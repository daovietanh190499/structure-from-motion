# Structure From Motion

# First image pair retrieval

First image pair is greatly affect the result of structure from motion pipeline. First image pair should have as many matching feature keypoints as possible \
and the repartition of the corresponding features in each image. But at the same time, the baseline or angle between cameras should also be large enough \
to provide reliable geometric information. In my experience, we can choose first image pair manually and put it on the first position of dataset
