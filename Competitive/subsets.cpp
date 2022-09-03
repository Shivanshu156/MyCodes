void subset(vector<int> &A, int l, int r, vector<vector<int>> &ans)
{       vector<int> a;
        if(l==r) { a.clear(); ans.push_back(a); a.push_back(A[l]); ans.push_back(a);   }
        else{
            
            subset(A, l+1, r, ans);
            int n = ans.size();
            for(int i=0;i<n;i++){
            vector<int> temp = ans[i];
            temp.push_back(A[l]);
            sort(temp.begin(), temp.end());
            ans.push_back(temp);
            }
        }


}
vector<vector<int> > Solution::subsets(vector<int> &A) {

    vector<vector<int>> ans;
    
    int l=0, r=A.size()-1;
    // cout<<" r is "<<r;
    if(l<=r)
    {subset(A, l , r, ans);
    sort(ans.begin(), ans.end());}
    else ans.push_back(A);
    return ans;
    
    
}