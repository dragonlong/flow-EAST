
### 1. Introduction & Problem statement 
As an important and challenging problem in computer vision, scene text detection has been drawing researchers' interest. The performance of text detection are largely pushed forward along with the boom of deep learning.
However, although people have proposed different models to improve text detection in single image, less attention is paid to improve text detection in video, which is more challenging due to effects like motion blur, extreme rotation of text lines. 

Given a video as input, we want to build a new model based on existing single image text detector, and improve the performance regarding accuracy, while not bring to much overhead to system efficiency. 

### 2. Flow Estimation  Test

<img src="images/flow_c.png" style="width:1200px;height:400px;">
<caption><center> <u> <font color='purple'> **Figure 1** </u><font color='purple'>  : **Flow Estimation**<br> From left to right are: image1, image2, dense flow map, warpped image1. Warpped image1 should be close to image2</center></caption>

### 3. Results Analysis on ICDAR 2013 Benchmark

<img src="images/good.png" style="width:900px;height:500px;">
<caption><center> <u> <font color='purple'> **Figure 2** </u><font color='purple'>  : **Good Examples**<br></center> Boxes that have challenging rotation angle, or small size, could be detected by the new model, while original EAST couldn't detect them very well, also the detected boxes geometry are more precise.</caption>

<img src="images/fail.png" style="width:900px;height:500px;">
<caption><center> <u> <font color='purple'> **Figure 3** </u><font color='purple'>  : **Failure Cases**<br> </center>For boxes sitting near the boundary, feature aggregation would sometimes fail due to imprecise flow esimation; robust flow estimation guarantees the precision for boxes prediction.</caption>

### 4. Preliminary Test on ICDAR 2013 Benchmark

<img src="images/comp.png" style="width:1000px;height:500px;">
<caption><center> <u> <font color='purple'> **Figure 4** </u><font color='purple'>  : **Results Comparison**<br> </center></caption>

### 5. Conclusion

From the preliminary test results, we could see the detection performance for some videos are boosting up when we apply flow-based feature aggregation to a single image text detector, the recall has significant improvement; However, the dense feture aggregation is not robust to all videos, especially for regions that flow estimation is not accurate. Further ways to improve flow estimation or reduce the dependency on flow need to be proposed. 
