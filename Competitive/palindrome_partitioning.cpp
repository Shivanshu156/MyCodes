// Given a string s, partition s such that every string of the partition is a palindrome.

// Return all possible palindrome partitioning of s.

// For example, given s = "aab",
// Return

//   [
//     ["a","a","b"]
//     ["aa","b"],
//   ]
//  Ordering the results in the answer : Entry i will come before Entry j if :
// len(Entryi[0]) < len(Entryj[0]) OR
// (len(Entryi[0]) == len(Entryj[0]) AND len(Entryi[1]) < len(Entryj[1])) OR
// *
// *
// *
// (len(Entryi[0]) == len(Entryj[0]) AND … len(Entryi[k] < len(Entryj[k]))
// In the given example,

bool ispalindrome(string A)
{   bool check = true; int n = A.size();
    for(int i=0;i<n/2;i++)
    if(A[i]!=A[n-i-1]) check = false;
    return check;
}

void pal(string A, int start, vector<string> &a, vector<vector<string>> &ans)
{
    if(start>=A.size()){ans.push_back(a); return;}
    
    for(int i=start;i<A.size();i++)
    {   if(ispalindrome(A.substr(start,i-start+1)))
        {   a.push_back(A.substr(start,i-start+1));
            pal(A,i+1,a,ans);
            a.pop_back();
        }
    }
}
vector<vector<string> > Solution::partition(string A) {
   
    vector<vector<string>> ans;
    vector<string> a;
    pal(A,0, a,ans);
    sort(ans.begin(), ans.end());
    return ans;
    
}