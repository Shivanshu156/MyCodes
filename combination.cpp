// Given two integers n and k, return all possible combinations of k numbers out of 1 2 3 ... n.

// Make sure the combinations are sorted.

// To elaborate,

// Within every entry, elements should be sorted. [1, 4] is a valid entry while [4, 1] is not.
// Entries should be sorted within themselves.
// Example :
// If n = 4 and k = 2, a solution is:

// [
//   [1,2],
//   [1,3],
//   [1,4],
//   [2,3],
//   [2,4],
//   [3,4],
// ]
//  Warning : DO NOT USE LIBRARY FUNCTION FOR GENERATING COMBINATIONS.
// Example : itertools.combinations in python.

void comb(int start, int n, int k, vector<int> &a, vector<vector<int>> &ans)
{
    if(a.size()==k){ ans.push_back(a); return;}
    for(int i=start+1; i<=n;i++)
    {   a.push_back(i);
        comb(i,n,k,a,ans);
        a.pop_back();
    }
}
vector<vector<int> > Solution::combine(int A, int B) {
    vector<vector<int>> ans;
    vector<int> a;
    comb(0, A, B, a, ans);
    sort(ans.begin(), ans.end());
    return ans;
}
