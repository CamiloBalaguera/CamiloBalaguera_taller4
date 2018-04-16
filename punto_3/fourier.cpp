#include <iostream>
using namespace std;
float interpolar (float x, double* puntosx, double* puntosy);
int main()
{
	double x[2];
	x[0] = 0;
	x[1] = 1;
 	double y[2];
	y[0] = 1;
	y[1] = 4;
	double in = 0.0;
	double fi = 3.0;
	double lista[100];
	double h = (fi-in)/100;
	for (int i = 0; i < 100; i++)
	{
		cout << i << " , " << interpolar(in + i*h, x, y) << endl;
	}
	return 0;
}			

float interpolar (float x, double* puntosx, double* puntosy)
{
	float px = 0.0;
	int n = sizeof(puntosx);
	for(int i = 0; i < n; i++)
	{
		float l = puntosy[i];
		for(int j = 0; j<n; j++)
		{
			if (j != i)
			{
				l = l *((x-puntosx[j])/(puntosx[i]-puntosx[j]));
				px += l;
			}
		}	
	}
	return px;	
}
