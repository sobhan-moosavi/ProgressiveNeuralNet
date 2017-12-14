
if not zoo then
    require 'initenv'
end

local npl, nl = torch.class('zoo.NeuralPLearner', 'zoo.NeuralLearner')


function npl:__init(args)
   nl.__init(self, args)
   
   -- Q-learning parameters
   self.clip_delta     = args.clip_delta     -- Whether to clip the temporal difference error.
   self.target_q       = args.target_q or 0  -- Whether to use a target network. (0 for none)

   -- Q-learning validation statistics
   self.v_avg     = {} -- Per-game V running average
   self.tderr_avg = {} -- Per-game TD error running average   
   self.v_avg[1]     = 0 -- We just have one game 
   self.tderr_avg[1] = 0 -- We just have one game   
	
   -- Load target network, if needed
   if self.target_q > 0 then
      self.target_network = self.network:clone()
   end
   
   -- Load Expert Nets: These are sourc games which will be used in progressive training approach   
   self.expertnet_prefix = args.expertnet_prefix
   self.source_games = args.source_games
   self.n_actions = 18
   self:loadExpertNets()
   
	--Make partial copies of source game related nets
	self.expnet_fc_1 = {}
	self.expnet_fc_2 = {}
	self.expnet_fc_b_1 = {}
	self.expnet_fc_b_2 = {}
	self.expnet_fc_w_1 = {}
	self.expnet_fc_w_2 = {}
	for i=1,#self.source_games do
		self.expnet_fc_1[i] = self:tmpCreateNet(self.expertnet[i], 1, 8) -- Size: 1 x 3,136
		self.expnet_fc_2[i] = self:tmpCreateNet(self.expertnet[i], 1, 10) -- Size: 1 x 512
		self.expnet_fc_b_1[i] = torch.mm(torch.ones(self.minibatch_size, 1), self:copyBias(self.expertnet[i]:get(9).bias:float())) -- Size: 32 x 512
		self.expnet_fc_b_2[i] = torch.mm(torch.ones(self.minibatch_size, 1), self:copyBias(self.expertnet[i]:get(11).bias:float())) -- Size: 32 x 18		
		self.expnet_fc_w_1[i] = self.expertnet[i]:get(9).weight:transpose(1,2):float() -- Size: 3,136 x 512	
		self.expnet_fc_w_2[i] = self.expertnet[i]:get(11).weight:transpose(1,2):float() -- Size: 512 x 18
	end
	
	self.pnet_b_1 = torch.Tensor(self.minibatch_size, self.network:get(9).bias:float():size(1)):fill(0) -- Size: 32 x 512
	self.pnet_b_2 = torch.Tensor(self.minibatch_size, self.network:get(11).bias:float():size(1)):fill(0) -- Size: 32 x 18
	
	--self.timer = torch.Timer()
	
	self.relu = nn.ReLU()
end

function npl:copyBias(x)
	local b = torch.Tensor(1, x:size(1))
	for i=1, x:size(1) do
		b[1][i] = x[i]
	end
	return b
end

function npl:loadExpertNets()
	self.expertnet = {}
	for i=1,#self.source_games do
		local netname = self.expertnet_prefix .. self.source_games[i] .. '.t7'
		collectgarbage()
		print('Loading Expert Network ' .. i .. ' from ' .. netname)
		local msg, exp = pcall(torch.load, netname)
		if not msg then
			error('Error loading expert network')
		end
		self.expertnet[i] = exp.model
		self.expertnet[i]:cuda()
		self.expertnet[i]:forward(torch.zeros(1,unpack(self.input_dims)):cuda())
		
		local actions = self:returnActions(self.source_games[i])
		
		local nl = #self.expertnet[i].modules
		local  l =  self.expertnet[i].modules[nl]
		
		-- If the expert network has only game-specific valid actions (<18) as output, convert it into a full 18-action output
		if l.output:size(2) < 18 then
			local newl = nn.Linear(l.weight:size(2), self.n_actions)
			newl.weight:zero()
			newl.bias:zero()

			for j=1,#actions do
				newl.weight[{{actions[j]},{}}]:copy(l.weight[{{j},{}}])
				newl.bias[actions[j]] = l.bias[j]
			end
			newl:cuda()

			-- Sanity check
			local valcount = 1
			for j=1,self.n_actions do
				if j == actions[valcount] then
					assert(torch.sum(torch.abs(torch.add(l.weight[valcount], -1, newl.weight[j]))) == 0)
					assert(math.abs(l.bias[valcount] - newl.bias[j]) == 0)
					valcount = valcount + 1
				else
					assert(torch.sum(torch.abs(newl.weight[j])) == 0)
					assert(newl.bias[j] == 0)
				end
			end
			self.expertnet[i].modules[nl] = newl
		end
		
	end
end

function npl:returnActions(game)	
	local actions = {}
	if game == 'asterix' then
		local _actions = {1, 3, 4, 5, 6, 7, 8, 9, 10}
		for i=1,#_actions do
			actions[i] = _actions[i]
		end
	elseif game == 'gopher' then 
		local _actions = {1, 2, 3, 4, 5, 11, 12, 13}
		for i=1,#_actions do
			actions[i] = _actions[i]
		end
	elseif game == 'breakout' then
		local _actions = {1, 2, 4, 5}
		for i=1,#_actions do
			actions[i] = _actions[i]
		end
	elseif game == 'pong' then
		local _actions = {1, 4, 5}
		for i=1,#_actions do
			actions[i] = _actions[i]
		end
	elseif game == 'bowling' then
		local _actions = {1, 2, 3, 4, 5, 6}
		for i=1,#_actions do
			actions[i] = _actions[i]
		end
	else
		for i=1,18 do
			actions[i] = i
		end
	end
	return actions
end

function npl:tmpCreateNet(net,s,e)
	local _net = nn.Sequential()
	for i=s,e do
		_net:add(net:get(i))
	end
	return _net
end

function npl:getNetworkTarget(args)
   local s2   = args.s2
   local mask = args.mask

   local target_q_net
   if self.target_q > 0 then
      target_q_net = self.target_network
   else
      target_q_net = self.network
   end

   -- Compute Q-learning next-step action value:
   -- max_a Q(s_2, a).
   local q2_full = target_q_net:forward(s2):float() -- DO WE NEED TO UPDATE THIS PART?
   
   -- Since we're masking out values, we don't want to select target action values that are not
   -- valid, so we fill all non-valid action-values with the minimum of the entire batch output
   -- This will ensure that a non-valid action value will not be chosen as a target incorrectly
   local fillVal = torch.min(q2_full)
   q2_full:maskedFill(torch.ne(mask,1), fillVal)
   self.q2_max = q2_full:max(2)

   return self.q2_max
end

function npl:getUpdate(args)

    local s, a, r, term
    local q, q2, q2_target
    local termnot

    s = args.s
    a = args.a
    r = args.r
    term = args.term
	
    -- calculate (1-terminal)
    termnot = term:clone():float():mul(-1):add(1)

    -- Compute the next-step action-value target
    q2_target = self:getNetworkTarget(args)

    -- Compute q2 = (1-terminal) * gamma * Q_target(s2,a') for some a' depending on method
    q2 = q2_target:clone():mul(self.discount):cmul(termnot)
	
    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    self.delta = r:clone():float()
    if self.rescale_r then
		local rangesize = q2:size(1) / self.n_games       
		local g_range = {{1,rangesize}} -- Just have one game
		self.delta[g_range]:div(self.r_max[1]) -- Just have one game 
    end
    self.delta:add(q2)
	
	-- PRogressive Network UPDATE
	local pnet_fc_1 = self:tmpCreateNet(self.network, 1, 8)
	local pnet_fc_2 = self:tmpCreateNet(self.network, 1, 10)
	local pnet_w_1 = self.network:get(9).weight:transpose(1,2):float() -- Size: 3,136 x 512	
	local pnet_w_2 = self.network:get(11).weight:transpose(1,2):float() -- Size: 512 x 18
	
	local FC1 = torch.Tensor(self.minibatch_size,512):fill(0)
	local FC2 = torch.Tensor(self.minibatch_size,18):fill(0)
	
	for i=1,#self.source_games do
		local M = torch.Tensor(self.minibatch_size,512):fill(0) 
		torch.mm(M, self.expnet_fc_1[i]:forward(s):float(), self.expnet_fc_w_1[i])
		torch.add(M, M, self.expnet_fc_b_1[i])
		torch.add(FC1, FC1, M)
		
		local M2 = torch.Tensor(self.minibatch_size,18):fill(0) 
		torch.mm(M2, self.expnet_fc_2[i]:forward(s):float(), self.expnet_fc_w_2[i])
		torch.add(M2, M2, self.expnet_fc_b_2[i])
		torch.add(FC2, FC2, M2)		
	end
	
	local P_FC1 = torch.Tensor(self.minibatch_size,512):fill(0)
	local P_FC2 = torch.Tensor(self.minibatch_size,18):fill(0)
	
	torch.mm(P_FC1, pnet_fc_1:forward(s):float(), pnet_w_1)
	torch.add(P_FC1, P_FC1, self.pnet_b_1)
	torch.add(P_FC1, P_FC1, FC1) --Add progressive connections
	P_FC1 = self.relu:forward(P_FC1)
	
	torch.mm(P_FC2, P_FC1, pnet_w_2)
	torch.add(P_FC2, P_FC2, self.pnet_b_2)
	torch.add(P_FC2, P_FC2, FC2) --Add progressive connections
	
	-- q = Q(s,a)
    local q_all = self.network:forward(s):float() -- Repalce this line with following line to train by Progressive approach
	q_all = P_FC2:float() -- I have to keep the above line for some dummy torch run-time error <some type mismatching>!! 
		
    q = torch.FloatTensor(q_all:size(1))		
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
		
    self.delta:add(-1, q)
	
    if self.clip_delta then
        self.delta[self.delta:ge(self.clip_delta)] = self.clip_delta
        self.delta[self.delta:le(-self.clip_delta)] = -self.clip_delta
		self.delta:add(0.0001)
    end

    local targets = q_all:clone():zero()
    for i=1,math.min(targets:size(1), a:size(1)) do
        targets[i][a[i]] = self.delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

	--print('new update: ' .. self.timer:time().real)
    return targets
end

function npl:perceive(rewards, rawstates, terminals, testing, testing_ep)
   actionIndices = nl.perceive(self, rewards, rawstates, terminals, testing, testing_ep)

   -- Have a chance to update the target q network every
   -- 'target_q' number of iterations
   if self.target_q > 0 and self.numSteps % self.target_q == 1 then
      self.target_network = nil
      collectgarbage()
      self.target_network = self.network:clone()
	  self.pnet_b_1 = torch.mm(torch.ones(self.minibatch_size, 1), self:copyBias(self.network:get(9).bias:float())) -- Size: 32 x 512
	  self.pnet_b_2 = torch.mm(torch.ones(self.minibatch_size, 1), self:copyBias(self.network:get(11).bias:float())) -- Size: 32 x 18
   end

   return actionIndices
end

function npl:compute_validation_statistics()
   --[[
   -- Convert the states to gpu, if necessary
   if self.gpu >= 0 then
      self.valid_s  = self.valid_s:cuda()
      self.valid_s2 = self.valid_s2:cuda()
   end
   
   local targets = self:getUpdate{s=self.valid_s, a=self.valid_a, r=self.valid_r, 
				   s2=self.valid_s2, a2=self.valid_a2, term=self.valid_t,
				   mask=self.val_mask}
   
   for i=1,self.n_games do
      g_range = {{(i-1)*self.valid_size+1, i*self.valid_size}}
      self.v_avg[i]     = self.q2_max[g_range]:mean()
      self.tderr_avg[i] = self.delta[ g_range]:clone():abs():mean()
   end
   
   -- Reconvert back to CPU RAM to save memory
   if self.gpu >= 0 then
      self.valid_s  = self.valid_s:float()
      self.valid_s2 = self.valid_s2:float()
   end
   --]]
end


function npl:print_validation_statistics()
   --[[
   print('Per-game validation statistics:')
   for i=1,self.n_games do
      print("\t" .. self.game_names[i] ..": V", self.v_avg[i], 
	    "TD error", self.tderr_avg[i], "Qmax", self.o_max[i])
   end
   --]]

end
