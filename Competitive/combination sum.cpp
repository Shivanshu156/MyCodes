// Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

// The same repeated number may be chosen from C unlimited number of times.

//  Note:
// All numbers (including target) will be positive integers.
// Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
// The combinations themselves must be sorted in ascending order.
// CombinationA > CombinationB iff (a1 > b1) OR (a1 = b1 AND a2 > b2) OR … (a1 = b1 AND a2 = b2 AND … ai = bi AND ai+1 > bi+1)
// The solution set must not contain duplicate combinations.
// Example,
// Given candidate set 2,3,6,7 and target 7,
// A solution set is:

// [2, 2, 3]
// [7]




void csum(vector<int> &A, int target, vector<vector<int>> &ans, vector<int> &a, int index)
{   int i = index;
    if(target == 0){ ans.push_back(a); return;}
    while(i<A.size() && target >= A[i])
    {               a.push_back(A[i]);
                    // cout<<A[i]<<endl;
                    csum(A,target-A[i],ans, a, i);  
                    i++;
                    
        a.pop_back();
    }
    
}
vector<vector<int> > Solution::combinationSum(vector<int> &A, int B) {
    vector<vector<int>> ans, new_ans;
    vector<int> a, A1;
    a.empty();
    sort(A.begin(), A.end());
    A1.push_back(A[0]) 
    int i = 1, j=0;
    
    while(i<A.size() && j<A.size())
    {
        if(A[i]!=A1[j]) {A1.push_back(A[i]); j++;}
        i++;
        
    }
    
    csum(A1,B,ans, a,0);
    sort(ans.begin(), ans.end());
    for(int i=0;i<ans.size();i++)
    if(ans[i].size()>1) new_ans.push_back(ans[i]);
    return ans;
}