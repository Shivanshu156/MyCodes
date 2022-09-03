#include <iostream>
using namespace std;

void swap(int *x, int*y)
{ int temp=*x;
      *y=*x;
      *x=temp;
}

void BubbleSort(int a[], int n)
{
	for(int i=0;i<n-1;i++)
	for(int j=0;j<n-i-1;j++)
	if(a[j]>a[j+1])
	swap(a[j],a[j+1]);
	
}

int main()
{   int n,arr[10]; 
	cout<<"Enter the size of the array";
	cin>>n;
    for (int i=0;i<n; i++)
    cin>>arr[i];
    BubbleSort(arr,n);
    cout<<"Sorted Array is: \t";
    for (int i=0;i<n;i++)
    cout<<arr[i]<<"  ";
    
}

