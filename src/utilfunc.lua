local M = {}

local meanim = torch.Tensor({129.67/255.0, 114.43/255.0, 107.26/255.0})

local scale = 4
local scales = {1/8,1/4,1/2,1}

function M.subtract_mean(im)

  im[{{},1,{},{}}] = im[{{},1,{},{}}] - meanim[1]
  im[{{},2,{},{}}] = im[{{},2,{},{}}] - meanim[2]
  im[{{},3,{},{}}] = im[{{},3,{},{}}] - meanim[3]
  return im

end

-- The input to the network in training is
-- {I1 - mean, I2 - mean}
-- And the output to the network in trainig is
-- (I2 - I1) * 100
--
-- Therefore we need the following preproces and
-- postprocess step
--
function M.preprocess(im1orig, im2orig)

  local nlen = im1orig:size(1)
  local h = im1orig:size(3)
  local w = im1orig:size(4)
  local im1 = {}
  local im2
  
  for i = 1, scale do
    im1[i] = torch.DoubleTensor(nlen, 3, torch.round(h * scales[i]), torch.round(w * scales[i]))
  end
  for i = 1, nlen do
    for j = 1, scale do 
      im1[j][i] = image.scale(im1orig[i], torch.round(scales[j] * w), torch.round(scales[j] * h))
    end
  end
  for j = 1, scale do 
    im1[j] = M.subtract_mean(im1[j])
  end
  if im2orig ~= nil then
    im2 = M.subtract_mean(im2orig:clone())
  else
    im2 = torch.DoubleTensor(nlen, 3, h, w)
  end

  if useCuda then
    for i = 1, scale do
      im1[i] = im1[i]:cuda()
    end
    im2 = im2:cuda()
  end

  input = im1
  table.insert(input, im2)
  return input

end

function M.postprocess(out, ref)

  return torch.add(out / 100, ref)

end

function M.getz(im1, im2, model)

  local batchsize = im1:size(1)

  local input = M.preprocess(im1, im2)
  model:forward(input)
  local z = model.encoder.output

  return z[1]
 
end

function M.forward(im, zsample, model)

  local batchsize = im:size(1)
  local seqmodel = model.seqmodel

  local input = M.preprocess(im) 
  model:forward(input)
  local mid1 = seqmodel:get(1):get(1).output
--  debugger:enter()
--  print(mid1[1][{1,1,1,{1,4},{1,4}}], mid1[2][{1,1,1,{1,4},{1,4}}],
--    mid1[3][{1,1,1,{1,4},{1,4}}], mid1[4][{1,1,1,{1,4},{1,4}}])
--  print(zsample[{1,{1,10}}])
  local mid2 = seqmodel:get(1):get(2):get(3):forward(zsample)
--  print(mid2[1][{1,1,1,1,{1,4},{1,4}}])
--  local mid2 = seqmodel:get(1):get(2):get(3):forward(zsample)
--  print(mid2[1][{1,1,1,1,{1,4},{1,4}}])
--  print(mid2[1][{1,1,1,1,{1,4},{1,4}}], mid2[2][{1,1,1,1,{1,4},{1,4}}],
--    mid2[3][{1,1,1,1,{1,4},{1,4}}], mid2[4][{1,1,1,1,{1,4},{1,4}}])
  local out = {mid1, mid2}
  for k = 2,6 do
    out = seqmodel:get(k):forward(out)
  end
  out = out:double()

  return M.postprocess(out, im):double()

end

function M.genGIF(infile, giffile, delay)
    if delay == nil then
        delay = 100
    end
    if giffile == nil then
        error('giffile empty')
    end
    local convertBinPath = '/data/vision/billf/motionTransfer/tfxue/tools/convert'
    os.execute(string.format('%s -delay %d %s %s', convertBinPath, delay, table.concat(infile, ' '), giffile))
end


return M

