// You are given an array A containing N integers. 
// The special product of each ith integer in this array is defined as the product of the following:<ul>

// LeftSpecialValue: For an index i, it is defined as the index j such that A[j]>A[i] (i>j). 
// If multiple A[j]’s are present in multiple positions, the LeftSpecialValue is the maximum value of j. 

// RightSpecialValue: For an index i, it is defined as the index j such that A[j]>A[i] (j>i). 
// If multiple A[j]s are present in multiple positions, the RightSpecialValue is the minimum value of j.

// Write a program to find the maximum special product of any integer in the array.

// Input: You will receive array of integers as argument to function.
// Return: Maximum special product of any integer in the array modulo 1000000007.
// Note: If j does not exist, the LeftSpecialValue and RightSpecialValue are considered to be 0.
// Constraints 1 <= N <= 10^5 1 <= A[i] <= 10^9

int Solution::maxSpecialProduct(vector<int> &A) {
	int n = A.size();
	vector<int> LeftSpecialValue(n,0),RightSpecialValue(n,0);
	stack<int> leftCalc;
	leftCalc.push(0);
	LeftSpecialValue[0]=0;
	for(int i=1;i<n;i++){
		while(!leftCalc.empty()){
			if(A[leftCalc.top()]>A[i]) break;
			leftCalc.pop();
		}
		LeftSpecialValue[i] = (leftCalc.empty())?0:leftCalc.top();
		leftCalc.push(i);
	}
	stack<int> rightCalc;
	rightCalc.push(n-1);
	RightSpecialValue[n-1]=0;
	for(int i=n-2;i>=0;i--){
		while(!rightCalc.empty()){
			if(A[rightCalc.top()]>A[i]) break;
			rightCalc.pop();
		}
		RightSpecialValue[i] = (rightCalc.empty())?0:rightCalc.top();
		rightCalc.push(i);
	}
	long long mx = -1;
	for(int i=0;i<n;i++){
		mx=max(mx,LeftSpecialValue[i]*1LL*RightSpecialValue[i]);
	}
	return mx%1000000007;
}