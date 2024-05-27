# VL53L8CX_ToF_Frames_Classification
Classifying frames obtained from an 8x8 ToF sensor (VL53L8CX) provided by STMicro-electronics.

The Aim of this project is to showcase the workflow of an end to end work using MATLAB (2024) and STM32Cube AI Developer cloud.

The frames are collected from the ToF sensor, which could record frames either in a 4x4 or 8x8 size. For the prupose of this study, we used the 8x8 setting.

For a better an optimal performance each frame (raw frame) is to be concatenated to its corresponding zonal frame, thus producing an 8x8x2 image (input size).
It is worth mentioning that both the raw and the zonal frames directly extracted from the VL53L8CX provided by STMicro-electronics

The workflow is chronological and involves a series of steps from preprocessing (the raw frames need to be brought to the proper range) and augmentations, to the benchmarking and deployment of the final model, which is a trained hyper parameters optimized, pruned and quantized model (for optimal efficiency and balance between flash and ram requirements during deployment).

Here is a brief description of the steps undertaken during this workflow:

- Preprocessing : Calibrating the raw frames into visible range, augmenting both the raw and zonal frames (12 Augmentations per frame) and concatenating a raw frame and its corresponding zonal frame.
  The Augmentations involve 12 transformations, aiming at enlarging the dataset (original size 5310 frames). The final dataset is has 69030 (8x8x2) frames.
  
- Building a classification Model: Since the frames are quite of small sizes, the original model suggested is a residual model with about 3.5K Learnables. This model architecture is useful to fight vanishing gradients.

- Checking Model Deployability: Using the STM32CubeAI Developer cloud to verify if the model's architecture is accepted by the boards used for inferencing. This makes it easier to proceed or to redesign the model in case it is not deployable.

- Training the Model: The model is trained using limited memory the BFGS algorithm, which is less costly than other optimizers like Adam or the sgdm.

- Deployability on STM32: The trained model accuracy is monitored, and the trained model is deployed on an STM32 Board on the STM32CubeAI Developer cloud.
  This step enables us to obtain the baseline of the upcoming models, observing the model's requirements (FLASH and RAM), as well as its MACC an Inference Time.
  
- Hyper Parameters Optimization: Having a good initial architecture, we aim at having (if possible) a smaller model which performs similarly or outperforms the original suggested model.
  This is done through an iterative optimization of the training parameters (Bayesian Optimization), by minimizing the objectve function, which is 1 - accuracy on the test set (convex optimization).
  
- Deployability on STM32: To measure how good this model performs, we do an inference on the STM32CUbeAI developer Cloud and measure its own requirements.
  The idea is to have less requirements from this model, since it should be a model with lesser or equal parameters as the originally suggested model.

- Pruning: The pruning step involves two steps. To avoid a series of gambles in order to get the best pruning percentage, we initially go through magnitude pruning.
  This gives us an estimation of the performance of the model for each percentage of sparsity introduced in the model.
  From the overall estimation provided by magnitude pruning, we can estimate the percentage of learnables we want to prune from the model (Using Taylor's Score)
  
- Deployability on STM32: The pruned model stands as a model performimg similarly to the previous ones, but with smaller parameters.
  This inferencing this model on the STM32Cube AI Developer cloud provides better performances (less MACC and a better Inference time).
  
- Qunatization: The model quantization step enables a low cost on board deployment. This could be done through QAT (quantization aware training) or PTQ (post training quantization).
  For this study, we used PTQ in MATLAB 2024 (8-bit post training quantization), which is quite optimized and produces excellent performances.
