clear all;
%% caffe environment initial
caffe.reset_all();
use_gpu = 1;  %  wheather using GPU ?
if (use_gpu)
    caffe.set_mode_gpu();
    gpu_id = 0;
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end
model_dir = '../model/';
net_model = [model_dir 'Imploy.prototxt'];
net_weights = [model_dir 'NOI_All-Iter_850000.caffemodel'];
phase = 'test';
if ~exist(net_weights, 'file')
    error('check the file is exist');
end
net = caffe.Net(net_model, net_weights, phase);

%% import testdata
load ImgDataTE.mat;

testWidth = 256;
testHeight = 256;



testimg = zeros(testWidth, testHeight, 1);

for i = 1:100
    testimg = ImgNoi15TE(:,:,i); 
    %imgdata = imresize(testimg, [input_height input_width], 'cubic');
    figure(1)
    subplot(2,1,1); imshow(testimg, [0/3000, 1200/3000]), title('srcimg');
    inputdata = {testimg};
    outimg = net.forward(inputdata);
    outdata = cell2mat(outimg);
    %outmat = imresize(outdata, [256 256], 'cubic');
    subplot(2,1,2);
    imshow(outdata, [0/3000, 1200/3000]), title('testimg');
    pause;
end

%% release caffe



