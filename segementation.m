% ***** loading the data *****

main_path = fullfile("daffodilSeg");
image_dir = fullfile(main_path,'ImagesRsz256');
pixel_dir = fullfile(main_path,'LabelsRsz256');

images = imageDatastore(image_dir,'ReadFcn', @(filename)imresize(imread(filename), [256 256]));


classes = ["daffodil","background"];
labels = [1,3];

pixels = pixelLabelDatastore(pixel_dir, classes , labels);

training_data = pixelLabelImageDatastore(images,pixels);


% ***** creating the model *****

inputSize = [256 256 3];

% initializing the graph
fcn = layerGraph;


% downsampling layers

down_sample_1 = [
    imageInputLayer(inputSize,'Name','image_input')
    convolution2dLayer(3,32,'Padding','same','Name','ds1_conv_1')
 
 batchNormalizationLayer('Name','ds1_bn_1')
  reluLayer("Name",'ds1_relu_1')

maxPooling2dLayer(2,'Stride',[2,2],'Name','ds1_mp_1')
];

down_sample_2 = [
    convolution2dLayer(3,64,'Padding','same','Name','ds2_conv_1')
 
 batchNormalizationLayer('Name','ds2_bn_1')
  reluLayer("Name",'ds2_relu_1')

maxPooling2dLayer(2,'Stride',[2,2],'Name','ds2_mp_1')
];


down_sample_3 = [
    convolution2dLayer(3,128,'Padding','same','Name','ds3_conv_1')
 
 batchNormalizationLayer('Name','ds3_bn_1')
  reluLayer("Name",'ds3_relu_1')

];


% upsampling: transposed convolution


up_sample_1 = [
    
 transposedConv2dLayer(2,64,"Stride",2,'Name','us1_tconv_1')

];


% depthwise concatenation node 1
depth_concat_1 = depthConcatenationLayer(2,'Name',"dc1");



% convolution after first upsampling
up_conv_1 = [
    

convolution2dLayer(3,64,'Padding','same','Name','uc1_conv_1')



];


% upsampling: transposed convolution 2
up_sample_2 = [
    
 transposedConv2dLayer(2,32,"Stride",2,'Name','us2_tconv_1')

];

% depthwise concatenation node 2

depth_concat_2 = depthConcatenationLayer(2,'Name',"dc2");

% convolution after second upsampling
up_conv_2 = [
    

convolution2dLayer(3,32,'Padding','same','Name','uc2_conv_1')



];

% final layer
final_layer = [
    
convolution2dLayer(1,2,"Name",'fl_conv1')
softmaxLayer
pixelClassificationLayer


];


% adding layers to the graph
fcn = addLayers(fcn,down_sample_1);

fcn = addLayers(fcn,down_sample_2);

fcn = addLayers(fcn,down_sample_3);

fcn = addLayers(fcn,up_sample_1);

fcn = addLayers(fcn,depth_concat_1);

fcn = addLayers(fcn,up_conv_1);


fcn = addLayers(fcn,up_sample_2);

fcn = addLayers(fcn,depth_concat_2);

fcn = addLayers(fcn,up_conv_2);

fcn = addLayers(fcn,final_layer);



% connecting the layers added to the graph

fcn = connectLayers(fcn,'ds1_mp_1','ds2_conv_1');


fcn = connectLayers(fcn,'ds2_mp_1','ds3_conv_1');

fcn = connectLayers(fcn,'ds3_relu_1','us1_tconv_1');

fcn = connectLayers(fcn,'us1_tconv_1','dc1/in1');

fcn = connectLayers(fcn,'ds2_relu_1','dc1/in2');

fcn = connectLayers(fcn,'dc1','uc1_conv_1');

fcn = connectLayers(fcn,'uc1_conv_1','us2_tconv_1');

fcn = connectLayers(fcn,'us2_tconv_1','dc2/in1');

fcn = connectLayers(fcn,'ds1_relu_1','dc2/in2');

fcn = connectLayers(fcn,'dc2','uc2_conv_1');

fcn = connectLayers(fcn,'uc2_conv_1','fl_conv1');


% training parameters
opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',10);


% model training
% net = trainNetwork(training_data,fcn,opts);


% Uncomment to analyse network
analyzeNetwork(fcn)

% Uncomment to save the trained model
% save('flower_segmentation.mat', 'net');


% ***** Uncomment to Evaluate the trained model *****


% test_path = fullfile("daffodil_test");
% test_image_dir = fullfile(test_path,'img');
% test_pixel_dir = fullfile(test_path,'pixel');
% 
% test_images = imageDatastore(test_image_dir,'ReadFcn', @(filename)imresize(imread(filename), [256 256]));
% test_pixels = pixelLabelDatastore(test_pixel_dir, classes , labels);
% 
% loaded_model = load('segmentnet.mat') ;
% predictions = semanticseg(...
% test_images,loaded_model.net, ...
%     'MiniBatchSize',5, ...
%     'WriteLocation',tempdir, ...
%     'Verbose',false);
% 
% 
% 
% metrics = evaluateSemanticSegmentation(predictions,test_pixels,'Verbose',false);
% 
% 
% iou_data = metrics.ClassMetrics.IoU;
% 
% disp(iou_data)
% 
% confusion_matrix = table2array(metrics.NormalizedConfusionMatrix);
% cm = confusionchart(int32(round(confusion_matrix*100,2)),classes);
% 
% histogram('Categories',{'Daffodil','Background'},'BinCounts',iou_data')



