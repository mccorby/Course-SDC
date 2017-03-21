#**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps each applied to the result of the previous one:
1. Convert the image into grayscale
2. Limit the region to be analyzed by selecting a triangle
3. Smooth the image using Gaussian Blur
4. Detect edges using Canny Edge Detector algorithm
  * Applied with values `low_threshold = 50` and `high_threshold = 150`
5. Applied Hough Transform with the following parameters
* `rho = 2`
  * `theta = 1 degree`
  * `threshold = 15 points`
  * `min_line_length = 40 pixels`
  * `max_line_gap = 20 pixels`
6. The lines obtain from the Hough Transform are then drawn using the `draw_lines()` function

In order to draw a single line on the left and right lanes, I modified the draw_lines() function as follows:
  * Curated the line set by removing those lines that are outliers. I assumed the distribution of the slopes is normal
  * The function will draw the line by using the mean slope and the mean center point of the curated set of lines

If you'd like to include images to show how the pipeline works, here is how to include an image:

![alt text][./test_images_output/solidWhiteCurce.jpg]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when a set of lines is not valid according to the curation process.
In this case the line is not drawn. This could be avoid by using global variables that would keep the values used in the previous frames.

Another shortcoming could be the non-selection by color and the possibility of selecting another area of interest as it can be seen in the Optional Challenge



###3. Suggest possible improvements to your pipeline

A possible improvement would be to allow a better definition of the parameters used in the pipeline, mainly those used by canny and hough transform. Having a different mechanism to pass them would allow to try with different configurations in an easier manner

Another potential improvement could be to keep the values used in previous frames so that they can be used to average the slope in the frame being processed

Another improvement would be using `np.polyfit` to generate the lines
