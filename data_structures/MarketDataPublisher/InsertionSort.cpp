#include <iostream>
using namespace std;


void InsertionSort(int a[], int n)
{   int key;
	for(int i=1;i<n;i++)
	{  key=a[i];
	   int j=i-1;
	   while(j>=0 && a[j]>key)
	   { a[j+1]=a[j];
	      j--;
	   }
	    a[j+1]=key;
		
	}
}

int main()
{   int n,arr[10]; 
	cout<<"Enter the size of the array";
	cin>>n;
    for (int i=0;i<n; i++)
    cin>>arr[i];
    InsertionSort(arr,n);
    cout<<"Sorted Array is: \t";
    for (int i=0;i<n;i++)
    cout<<arr[i]<<"  ";
    
}

