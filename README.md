# forgery-detection-and-source-identification
Last couple for days I started to familiarize myself more with the Deep Learning techniques, Convolutional neural networks (CNNs) and stuff. It's fascinating how we can integrate these with Digital Forensics by creating various tools. The major applications that interested me is Forgery Detection and Source Identification. 

CNNs can be used to detect various types of image forgeries, such as copy-move forgery, where a part of an image is copied and pasted onto another part to conceal information. By training a CNN on a dataset of authentic and manipulated images, the network can learn to differentiate between them based on subtle differences in pixel values and textures.

And in case of Source Identification, each camera has a unique image signature due to factors like sensor noise, lens distortion, and other imperfections. By training a CNN on a dataset of images from different cameras, the network can learn to recognize these signatures and infer the source camera of a given image. 

So why CNN? 
Overall I believe CNNs can be used to build powerful tools for image analysis in digital forensics, offering high accuracy and efficiency in detecting image manipulations and identifying the source of images. 

Technical Stack - TensorFlow, Keras, scikit-learn and NumPy libraries.
PS: The NumPy array is filled with random values here which represents 100 images of sizes 128x128. 

