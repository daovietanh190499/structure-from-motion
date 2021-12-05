# Structure From Motion

# First image pair retrieval

First image pair is greatly affect the result of structure from motion pipeline. First image pair should have as many matching feature keypoints as possible and the repartition of the corresponding features in each image. But at the same time, the baseline or angle between cameras should also be large enough to provide reliable geometric information. In my experience, we can choose first image pair manually and put it on the first position of dataset. As the images below, we have 5 type of image pairs:

## Type 1
Two cameras have same location and same direction but different field of view. This view pair position is not able to be used for reconstructing 3d points but it's able to be used for estimates the intrinsic params. this view pair is also called "pure rotation". <br><br>
![View1](https://github.com/daovietanh190499/structure-from-motion/blob/main/view_pairs/view0.png "View1")
## Type 2
Two cameras have same location and same field of view but different direction. This view pair position is able to be used for reconstructing 3d points but the 3d points have very low quality. <br><br>
![View2](https://github.com/daovietanh190499/structure-from-motion/blob/main/view_pairs/view1.png "View2")
## Type 3 and Type 4
Two cameras have same direction and same field of view but different location. This view pair position is good to be used for reconstructing 3d points. It's the most popular position when the cameras move in the straight direction<br><br>
![View3](https://github.com/daovietanh190499/structure-from-motion/blob/main/view_pairs/view2.png "View3")
![View4](https://github.com/daovietanh190499/structure-from-motion/blob/main/view_pairs/view3.png "View4")
## Type 5
Two cameras have different location, different direction and same field of view. This view pair position is the best to be used for reconstructing 3d points. It has large baseline between cameras, the cameras also have enough feature points for reconstruct reliable 3d point cloud. <br><br>
![View4](https://github.com/daovietanh190499/structure-from-motion/blob/main/view_pairs/view4.png "View4")

# Test

To run code we have to change the img_dir = '../vkist2/' to your image direction and run the sfm.py file


python sfm.py


# My result

![Result1](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res1.png "Result1")
![Result2](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res2.png "Result2")
![Result3](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res3.png "Result3")
![Result4](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res4.png "Result4")
![Result5](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res5.png "Result5")
![Result6](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res6.png "Result6")
![Result7](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res7.png "Result7")
![Result8](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res8.png "Result8")
![Result9](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res9.png "Result9")
![Result10](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res10.png "Result10")
![Result11](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res11.png "Result11")
![Result12](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res12.png "Result12")
![Result13](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res13.png "Result13")
![Result14](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res14.png "Result14")
![Result15](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res15.png "Result15")
![Result16](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res16.png "Result16")
![Result17](https://github.com/daovietanh190499/structure-from-motion/blob/main/result/res17.png "Result17")
