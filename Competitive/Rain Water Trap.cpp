int Solution::trap(const vector<int> &A) {
    
    int n = A.size(), left_max=A[0], right_max=INT_MIN, ans = 0;
    
    for(int i=1;i<n;i++)
    right_max = max(right_max, A[i]);
    
    for(int i=1;i<n-1;i++)
    {
        if(left_max > A[i] && right_max >A[i])
        ans  += min(right_max, left_max) - A[i];
        
        if(A[i]> left_max) left_max = A[i];
        if(A[i]== right_max)
        {   right_max = 0; for(int j=i+1;j<n;j++) right_max = max(right_max, A[j]); }
        
    }
    
    return ans;
}



// O(n)


class Solution {
    public:
        int trap(int A[], int n) {
            int left = 0; int right = n - 1;
            int res = 0;
            int maxleft = 0, maxright = 0;
            while(left <= right){
                if(A[left] <= A[right]){
                    if(A[left] >= maxleft) maxleft = A[left];
                    else res += maxleft-A[left];
                    left++;
                }
                else{
                    if(A[right] >= maxright) maxright = A[right];
                    else res += maxright - A[right];
                    right--;
                }
            }
            return res;
        }
};
