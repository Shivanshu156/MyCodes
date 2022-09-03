// Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses of length 2*n.

// For example, given n = 3, a solution set is:

// "((()))", "(()())", "(())()", "()(())", "()()()"
// Make sure the returned list of strings are sorted.




void genPar(int n, int open, int close, string str, vector<string> &ans)
{   if(close == n)
    { ans.push_back(str);
        return;
    }
    else{
        if(open<n)
        {
            str += '(';
            genPar(n, open+1, close, str, ans);
            str.pop_back();
        }
        
        if(close<open)
        {
            str+=')';
            genPar(n, open, close+1, str, ans);
            str.pop_back();
        }
    }
}
vector<string> Solution::generateParenthesis(int A) {
    vector<string> ans;
    string str = "";
    genPar(A, 0, 0, str, ans);
    return ans;
}
