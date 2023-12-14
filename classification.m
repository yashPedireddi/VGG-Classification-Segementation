% ***** Loading the data *****

dataset_path = "./flower_classification";

% setting up augmentation parameters

image_data_augmentation = imageDataAugmenter(...
    'RandRotation',[-10 10],...
    'RandYReflection',1) ;

flower_dataset = imageDatastore(dataset_path,'IncludeSubfolders',true,'LabelSource','foldernames');
targetSize = [256,256,3];


[flower_dataset_train,flower_dataset_validation,flower_dataset_test] = splitEachLabel(flower_dataset,0.7, 0.2, 0.1,'randomized');


flower_dataset_train_augmented = augmentedImageDatastore(targetSize,flower_dataset_train,'DataAugmentation',image_data_augmentation);

flower_dataset_validation_augmented = augmentedImageDatastore(targetSize,flower_dataset_validation);

flower_dataset_test_augmented = augmentedImageDatastore(targetSize,flower_dataset_test);


% ***** Creating the Model Architecture *****

inputSize = [256 256 3];

% initializing the graph
cnn = layerGraph;


% main layer
main_layer_1 = [ 

 imageInputLayer(inputSize,'Name','image_input')
 convolution2dLayer(3,32,'Stride',[2,2],'Padding','same','Name','ml_conv_1')
 batchNormalizationLayer('Name','ml_bn_1')
 reluLayer("Name",'ml_relu_1')   
 maxPooling2dLayer(2,'Stride',[2,2],'Name','ml_mp_1')



];

% hidden layers

hidden_layer_1 = [

 convolution2dLayer(3,64,'Padding','same','Name','hl1_conv_1')
 
 batchNormalizationLayer('Name','hl1_bn_1')
  reluLayer("Name",'hl1_relu_1')

 convolution2dLayer(3,64,'Padding','same','Name','hl1_conv_2')
 
 batchNormalizationLayer('Name','hl1_bn_2')
  reluLayer("Name",'hl1_relu_2')
];

hidden_layer_2 = [

 convolution2dLayer(3,256,'Padding','same','Name','hl2_conv_1')
 
 batchNormalizationLayer('Name','hl2_bn_1')
  reluLayer("Name",'hl2_relu_1')

 convolution2dLayer(3,256,'Padding','same','Name','hl2_conv_2')

 batchNormalizationLayer('Name','hl2_bn_2')
  reluLayer("Name",'hl2_relu_2')

];

hidden_layer_3 = [

 convolution2dLayer(3,256,'Padding','same','Name','hl3_conv_1')
 
 batchNormalizationLayer('Name','hl3_bn_1')
  reluLayer("Name",'hl3_relu_1')

 convolution2dLayer(3,256,'Padding','same','Name','hl3_conv_2')
 
 batchNormalizationLayer('Name','hl3_bn_2')
  reluLayer("Name",'hl3_relu_2')
];


% Skip Connection Layers


skip_layer_1 = [

    convolution2dLayer(1,64,'Padding','same','Name','sl1_conv_1')
    batchNormalizationLayer('Name','sl1_bn_1')
    


    

];

skip_layer_2 = [

    convolution2dLayer(1,256,'Padding','same','Name','sl2_conv_1')
    batchNormalizationLayer('Name','sl2_bn_1')


    

];

skip_layer_3 = [

    convolution2dLayer(1,256,'Padding','same','Name','sl3_conv_1')
    batchNormalizationLayer('Name','sl3_bn_1')


    

];


% Adding operation nodes


add_1 =[

additionLayer(2,'Name',"add1")
    ];

add_2 =[

additionLayer(2,'Name',"add2")
    ];

add_3 =[

additionLayer(2,'Name',"add3")
    ];


% final layer

final_layer = [

globalAveragePooling2dLayer("Name",'fl_gap')
    fullyConnectedLayer(256)
    fullyConnectedLayer(17)
    softmaxLayer
    classificationLayer
    
];


% adding the created layers to the layer graph

cnn = addLayers(cnn,main_layer_1);
cnn = addLayers(cnn,hidden_layer_1 ) ;

cnn = addLayers(cnn, skip_layer_1 ) ;
cnn = addLayers(cnn, add_1 );

cnn = addLayers(cnn,hidden_layer_2 ) ;

cnn = addLayers(cnn, skip_layer_2 ) ;
cnn = addLayers(cnn, add_2 );

cnn = addLayers(cnn,hidden_layer_3 ) ;

cnn = addLayers(cnn, skip_layer_3 ) ;
cnn = addLayers(cnn, add_3 );

cnn = addLayers(cnn, final_layer ) ;



% implementing the connections between the layers

cnn = connectLayers(cnn,'ml_mp_1','hl1_conv_1');
cnn = connectLayers(cnn,'ml_mp_1','sl1_conv_1');
cnn = connectLayers(cnn,'hl1_relu_2','add1/in1');

cnn = connectLayers(cnn,'sl1_bn_1','add1/in2');

cnn = connectLayers(cnn,'add1','hl2_conv_1');
cnn = connectLayers(cnn,'add1','sl2_conv_1');

cnn = connectLayers(cnn,'hl2_relu_2','add2/in1');
cnn = connectLayers(cnn,'sl2_bn_1','add2/in2');


cnn = connectLayers(cnn,'add2','hl3_conv_1');
cnn = connectLayers(cnn,'add2','sl3_conv_1');

cnn = connectLayers(cnn,'hl3_relu_2','add3/in1');
cnn = connectLayers(cnn,'sl3_bn_1','add3/in2');

cnn = connectLayers(cnn,'add3','fl_gap');


% initializing training parameters

training_settings = trainingOptions('adam','MaxEpochs',100, ...
    'ValidationData',flower_dataset_validation_augmented, ...
    'ValidationFrequency',20, ...
    'InitialLearnRate',0.001,...
    'LearnRateDropFactor', 0.2000, ...
    'LearnRateDropPeriod', 20, ...
    'Verbose',false, ...
    'MiniBatchSize',32,...
    'Shuffle','every-epoch',...
    'Plots','training-progress');

% Training the model
net = trainNetwork(flower_dataset_train_augmented,cnn,training_settings);


% Uncomment below line to analyse the network
% analyzeNetwork(cnn)


% Saving the trained model
% save('classnet.mat', 'net');


% ***** Uncomment to run model evaluation *****

% model = load('classnet.mat') ;
% 
% 
% labelCategories = model.net.Layers(end).Classes ;
% 
% predictions = classify(model.net , flower_dataset_test_augmented ) ;
% 
% predictedLabels = labelCategories(predictions) ;
% 
% confusion_matrix_plot = confusionchart(labelCategories(flower_dataset_test.Labels), predictedLabels );


