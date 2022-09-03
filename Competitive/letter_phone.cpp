// Given a digit string, return all possible letter combinations that the number could represent.

// A mapping of digit to letters (just like on the telephone buttons) is given below.



// The digit 0 maps to 0 itself.
// The digit 1 maps to 1 itself.

// Input: Digit string "23"
// Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
// Make sure the returned strings are lexicographically sorted.


unordered_map<char, string> S;
void funct(string A, int start, string &a, vector<string> &ans)
{
    if(start>=A.size()) {ans.push_back(a); return;}
    string s = S[A[start]];
    for(int i=0;i<s.size();i++)
    {   string b = a;
        b+=s[i];
        funct(A, start+1, b, ans);
    }
}
vector<string> Solution::letterCombinations(string A) {
    S.clear();
    S.insert(make_pair('0',"0")); S.insert(make_pair('1',"1")); S.insert(make_pair('2',"abc"));
    S.insert(make_pair('3',"def")); S.insert(make_pair('4',"ghi")); S.insert(make_pair('5',"jkl"));
    S.insert(make_pair('6',"mno")); S.insert(make_pair('7',"pqrs")); S.insert(make_pair('8',"tuv"));
    S.insert(make_pair('9',"wxyz"));
    vector<string> ans;
    string a="";
    funct(A, 0, a, ans);
    sort(ans.begin(), ans.end());
    return ans;
}



