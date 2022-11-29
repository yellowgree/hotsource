  	
import cv
import cxcore
import highgui

IplImage*     g_image   = NULL;
IplImage*     g_gray    = NULL;
CvMemStorage*   g_storage = NULL;

int main( int argc, char** argv )
{
  g_image = cvLoadImage("text1-1.jpg");
  g_gray = cvLoadImage("text1-2.jpg");

  cvNamedWindow( "cvAbsDiff", 1 );
  cvNamedWindow( "Result", 1 );


  cvAbsDiff(g_image, g_gray, g_image);
  cvThreshold(g_image, g_image, 55, 255, CV_THRESH_BINARY);

  cvShowImage( "cvAbsDiff", g_image );

  g_gray = cvCreateImage( cvGetSize(g_image), 8, 1 );
  g_storage = cvCreateMemStorage(0);

  CvSeq* contours = 0;
  cvCvtColor( g_image, g_gray, CV_BGR2GRAY );
  cvFindContours( g_gray, g_storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

  g_image = cvLoadImage("C:\\1.bmp");

  if( contours ){
      cvDrawContours( g_image, contours, CV_RGB(255, 255, 0), CV_RGB(0,255,0), 100, -1,CV_AA );
  }

  cvShowImage( "Result", g_image );

  cvWaitKey();

  cvReleaseImage( &g_image );
  cvReleaseImage( &g_gray );
  cvReleaseMemStorage( &g_storage );
  cvDestroyWindow( "cvAbsDiff" );
  cvDestroyWindow( "Result" );
  return 0;
}
