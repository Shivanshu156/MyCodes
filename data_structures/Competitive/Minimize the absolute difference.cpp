// Given three sorted arrays A, B and Cof not necessarily same sizes.

// Calculate the minimum absolute difference between the maximum and minimum number from the triplet a, b, c such that a, b, c belongs arrays A, B, C respectively.
// i.e. minimize | max(a,b,c) - min(a,b,c) |.

// Example :

// Input:

// A : [ 1, 4, 5, 8, 10 ]
// B : [ 6, 9, 15 ]
// C : [ 2, 3, 6, 6 ]
// Output:

// 1
// Explanation: We get the minimum difference for a=5, b=6, c=6 as | max(a,b,c) - min(a,b,c) | = |6-5| = 1.



int Solution::solve(vector<int> &A, vector<int> &B, vector<int> &C) {
    int a, b, c, m=A.size(), n=B.size(), p=C.size(), i=0,j=0,k=0, ans = INT_MAX, count=0;
     
    
    while(i<m && j<n && k<p)
    {   a = A[m-1-i]; b = B[n-1-j]; c = C[p-1-k];
        ans = min(ans, max(max(a,b),c) - min(min(a,b),c));
        bool done = false;
        if(a==max(max(a,b),c) && !done)
        {   
            // cout<<"check1"<<endl;
            a = A[m-1-i];
            i++;
            ans = min(ans, max(max(a,b),c) - min(min(a,b),c));
            done = true;
        }
        if(b==max(max(a,b),c)&& !done)
        {   
            // cout<<"check2"<<endl;
            b = B[n-1-i];
            j++;
            ans = min(ans, max(max(a,b),c) - min(min(a,b),c));
            done = true;
        }
        if(c==max(max(a,b),c)&& !done)
        {   
            // cout<<"check3"<<endl;
            c = C[p-1-i];
            k++;
            ans = min(ans, max(max(a,b),c) - min(min(a,b),c));
            done = true;
        }
        
    }

    return ans;
}
