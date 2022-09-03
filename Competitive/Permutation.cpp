void swap(vector<int> &A,int l, int r )
{   int temp = A[l];
    A[l] = A[r];
    A[r] = temp;
}

void permutation(vector<int> &A, int l, int r, vector<vector<int>> &ans)
{   
    if(l==r) ans.push_back(A);
    for(int i=l; i<=r;i++)
    {
        swap(A,l,i);
        permutation(A,l+1,r, ans);
        swap(A,l,i);
        
    }
    
    
}

vector<vector<int> > Solution::permute(vector<int> &A) {
    vector<vector<int>> ans;
    int l=0, r = A.size()-1, n=A.size();
    permutation(A, l, r, ans);
    return ans;
    
}