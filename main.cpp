#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;

const String window_capture_name = "Video Capture";
const String window_detection_name = "Computation";

int currentAmount = 0;
int previousAmount = 0;
bool firstCheck = true;

Scalar colorOrange = Scalar(0, 153, 255);

int countPips(Mat dice) 
{
	// resize
	cv::resize(dice, dice, cv::Size(150, 150));

	dice.convertTo(dice, -1, 1, 150);			//brighten up the original image
	cvtColor(dice, dice, COLOR_BGR2GRAY);		//create a grayscale version of the image
	blur(dice, dice, Size(4, 4));		//blur the grayscale to reduce noise
	threshold(dice, dice, 150, 255, THRESH_OTSU);

	cv::imshow(window_detection_name, dice);

	// floodfill
	cv::floodFill(dice, cv::Point(0, 0), cv::Scalar(255));
	cv::floodFill(dice, cv::Point(0, 149), cv::Scalar(255));
	cv::floodFill(dice, cv::Point(149, 0), cv::Scalar(255));
	cv::floodFill(dice, cv::Point(149, 149), cv::Scalar(255));

	// search for blobs
	cv::SimpleBlobDetector::Params params;

	// filter by interia defines how elongated a shape is.
	params.filterByInertia = true;
	params.minInertiaRatio = 0.5;

	// will hold our keyponts
	std::vector<cv::KeyPoint> keypoints;

	// create new blob detector with our parameters
	cv::Ptr<cv::SimpleBlobDetector> blobDetector = cv::SimpleBlobDetector::create(params);

	// detect blobs
	blobDetector->detect(dice, keypoints);

	
	// return number of pips
	return (int)keypoints.size();

}

int main(int argc, char* argv[])
{
	//Initialize the webcamera and two screens
	VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
	namedWindow(window_capture_name, true);
	namedWindow(window_detection_name, true);

	//create the image that will be edited
	Mat image;

	while (true) 
	{
		//read the image that the webcam is capturing
		cap >> image;
		
		if (image.empty())
		{
			break;
		}
	
		Mat unprocessFrame = image.clone();

		image.convertTo(image, -1, 1, 150);			//brighten up the original image
		cvtColor(image, image, COLOR_BGR2GRAY);		//create a grayscale version of the image
		blur(image, image, Size(5, 5));		//blur the grayscale to reduce noise
		threshold(image, image, 150, 255,  THRESH_OTSU);
		//Canny(image, image, 2, 4, 3, false); //now turn the image either black or white

		std::vector<std::vector<Point>> diceContours;
		std::vector<Vec4i> diceHierarchy;
		//using the findContours function, it transforms the previously detected edges into a list of contours,
		//where each contour is a list of points that connected together form an edge;
		findContours(image.clone(), diceContours, diceHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		std::vector<RotatedRect> diceRects;

		for (int i = 0; i < diceContours.size(); i++)
		{
			//for each contour, we search the minimum area rectangle and enclose all the points of each contours 
			RotatedRect rect = minAreaRect(diceContours[i]);

			// Process only rectangles that are almost square and of the right size.
			double aspect = fabs(rect.size.aspectRatio() - 1);
			if ((aspect < 0.25) && (rect.size.area() > 2000) && (rect.size.area() < 4000))
			{
				
				bool process = true;
				for (int j = 0; j < diceRects.size(); j++)
				{
					double dist = norm(rect.center - diceRects[j].center);
					if (dist < 10) {
						process = false;
						break;
					}
				}

				if (process) 
				{
					diceRects.push_back(rect);


					Rect diceBoundsRect = boundingRect(Mat(diceContours[i]));

					Mat diceROI = unprocessFrame(diceBoundsRect);

					int numberOfPips = countPips(diceROI);

					if (numberOfPips > 0) {

						// ouput debug info
						std::ostringstream diceText;
						diceText << "val: " << numberOfPips;

						// draw value
						cv::putText(unprocessFrame, diceText.str(),
							Point(diceBoundsRect.x, diceBoundsRect.y + diceBoundsRect.height + 20),
							FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar::all(255), 1, 8
						);

						// draw bounding rect
						
						rectangle(unprocessFrame, diceBoundsRect.tl(), diceBoundsRect.br(), colorOrange, 2, 8, 0);

						if (firstCheck)
							currentAmount += numberOfPips;
						else
							previousAmount -= numberOfPips;
					}
				}
			}
		}

		std::ostringstream totalText;
		totalText << "Total: " << currentAmount;

		putText(unprocessFrame, totalText.str(),
			Point(0, 50), FONT_HERSHEY_TRIPLEX, 1.8, colorOrange, 3, 8);

		//total point counting update
		if (!firstCheck && previousAmount != 0)
		{
			currentAmount = 0;
			firstCheck = true;	
		}
		else
		{	
			firstCheck = false;
		}
		previousAmount = currentAmount;

		imshow(window_capture_name, unprocessFrame);
		imshow("Image", image);
		
		char key = (char)waitKey(10);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}
	return 0;
}