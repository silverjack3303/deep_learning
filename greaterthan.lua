require 'nn'

network=nn.Linear(2,1)

criterion = nn.MSECriterion()  

for i = 1,5000 do
  -- random sample
  local input= torch.randn(2);     -- normally distributed example in 2d
  local output= torch.Tensor(1);
  if input[1]-input[2] > 0 then  -- calculate label for XOR function
    output[1] = 1
  else
    output[1] = -1
  end

  -- feed it to the neural network and the criterion
  criterion:forward(network:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  network:zeroGradParameters()
  -- (2) accumulate gradients
  network:backward(input, criterion:backward(network.output, output))
  -- (3) update parameters with a 0.01 learning rate
  network:updateParameters(0.01)
end

print(network)
print(network.weight)

x = torch.Tensor(2)
x[1] =  1; x[2] =  0; print(network:forward(x))
x[1] =  0; x[2] =  1; print(network:forward(x))
x[1] =  10; x[2] =  0; print(network:forward(x))
x[1] =  10; x[2] =  100; print(network:forward(x))
x[1] =  10; x[2] =  -10; print(network:forward(x))
