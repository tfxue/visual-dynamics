-------------------- Options --------------------------------------
useCuda = true
gpuId = 1           -- GPU device ID
demo = 'all'        -- 'all', 'demo1', 'demo2', 'demo3'
modeldir = '../models/'
datadir = '../data/'
outdirRoot = '../output/'
createGIF = true -- Set this to true if you have installed ImageMagick
                 -- http://git.imagemagick.org/repos/ImageMagick

-------------------- Initialization --------------------------------------
print '==> Initializing...'
require 'image'
if useCuda then require 'cutorch' end
require 'nn'
if useCuda then require 'cudnn' end
if useCuda then require 'cunn' end
require 'layers/nnlayer_ext'
ut = require 'utilfunc'

if useCuda then
  cutorch.setDevice(gpuId)
end
if not path.isdir(outdirRoot) then
  path.mkdir(outdirRoot)
end

imsize = 128        -- All input image should be 128x128
model = torch.load(path.join(modeldir, 'exercise128_model.t7'))    -- Load model
zTrain = torch.load(path.join(modeldir, 'exercise128_z.t7'))
model:evaluate()
model.seqmodel:evaluate()

function loadIm(impath)
  
  local im = torch.reshape(image.scale(image.load(impath), imsize, imsize),
    1, 3, imsize, imsize);
  return im

end

----------------- Test 1: sample future frames (exercise) ---------------------------

if demo == 'all' or demo == 'demo1' then
  
  outdir = path.join(outdirRoot, 'demo1')
  if not path.isdir(outdir) then
    path.mkdir(outdir)
  end

  nsample = 10         -- Number of samples generated for each input image
  inputImPath = path.join(datadir, 'exercise_input.png')
  im = loadIm(inputImPath)
  zdim = zTrain:size(2)
  if useCuda then
    zTrain = zTrain:cuda()
    model.encoder = model.encoder:cuda()
    model.seqmodel = model.seqmodel:cuda()
  end
    -- In this demo, we only use 1000 randomly selected samples in training
    -- to approximate the empirical distribution of z in training

  nzTrain = zTrain:size(1)
  I1path = path.join(outdir, 'input.png')
  image.save(I1path, im[1])
  for i = 1,nsample do
    randidx = math.random(1,zTrain:size(1))
    zsample = torch.reshape(zTrain[randidx], 1, zdim)
    out = ut.forward(im, zsample, model)
    I2path = path.join(outdir, string.format('sample_%d.png', i))
    image.save(I2path, out[1])
    if createGIF then
      ut.genGIF({I1path, I2path}, path.join(outdir, 
        string.format('sample_%d.gif', i)))
    end
  end
  print('demo1 is done. Please check the folder \'../output/demo1\'')

end


----------------- Test 2: transfer motion from a source pair to a target --------------

if demo == 'all' or demo == 'demo2' then

  outdir = path.join(outdirRoot, 'demo2')
  if not path.isdir(outdir) then
    path.mkdir(outdir)
  end

  -- Load source
  source = {}
  source[1] = loadIm(path.join(datadir, 'sample1_im1.png'))
  source[2] = loadIm(path.join(datadir, 'sample1_im2.png'))
  sourcefiles = {}
  for i = 1, 2 do
    sourcefiles[i] = path.join(outdir, string.format('source_im%d.png',i))
    image.save(sourcefiles[i], source[i][1])
  end
  if createGIF then
    ut.genGIF(sourcefiles, path.join(outdir, 'source.gif'))
  end

  -- Get the motion representation of source
  z = ut.getz(source[1], source[2], model):clone()

  -- Synthesize new motion
  for i = 1, 2 do
    target = {}
    target[1] = loadIm(path.join(datadir, string.format('sample%d_im1.png', i+1)))
    target[2] = ut.forward(target[1], z, model)
    targetfiles = {}
    for j = 1, 2 do
      targetfiles[j] = path.join(outdir, string.format('target%d_im%d.png',i,j))
      image.save(targetfiles[j], target[j][1])
    end
    if createGIF then
      ut.genGIF(targetfiles, path.join(outdir, string.format('target%d.gif',i)))
    end
  end
  print('demo2 is done. Please check the folder \'../output/demo2\'')

end

----------------- Test 3: visualize the latent vector z ----------------------------

if demo == 'all' or demo == 'demo3' then

  outdir = path.join(outdirRoot, 'demo3')
  if not path.isdir(outdir) then
    path.mkdir(outdir)
  end

  selectedDimensions = {752, 1746, 2195}

  im1 = loadIm(path.join(datadir, 'sample1_im1.png'))
  im2 = loadIm(path.join(datadir, 'sample1_im2.png'))
  zorig = ut.getz(im1, im2, model):clone()
  newZvalue = torch.linspace(-10,10,21)
  
  for i,selectdim in pairs(selectedDimensions) do
   
    -- Forward 
    nz = newZvalue:size(1)
    if useCuda then
      out = torch.CudaTensor(nz, 3, imsize, imsize)
    else
      out = torch.DoubleTensor(nz, 3, imsize, imsize)
    end
    zmean = zorig[{1,selectdim}]
    for j = 1, nz do
      z = zorig:clone()
      z[{1,selectdim}] = newZvalue[j] + zmean
      out[{{j},{},{},{}}] = ut.forward(im1, z, model)
    end

    -- Save
    filelist = {}
    for j = 1, out:size(1) do
      filelist[j] = path.join(outdir,
        string.format('out_newz_dim%04d_%02d.png', selectdim, j))
      image.save(filelist[j], out[j]:double())
    end
    if createGIF then
      ut.genGIF(filelist, path.join(outdir,
        string.format('out_newz_dim%04d.gif', selectdim)), 10) 
    end
    -- print(string.format('dimension %04d is done', selectdim))
  end

  print('demo3 is done. Please check the folder \'../output/demo3\'')
end







