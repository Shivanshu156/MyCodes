// Given a list of non negative integers, arrange them such that they form the largest number.

// For example:

// Given [3, 30, 34, 5, 9], the largest formed number is 9534330.

// Note: The result may be very large, so you need to return a string instead of an integer.

bool compare(string a, string b)
{   return a+b > b+a;   }

string Solution::largestNumber(const vector<int> &A) {
    vector<string> Arr;
    string ans; bool check = false;
    
    for (int i=0; i<A.size(); i++)
    {if(A[i]!=0) check= true;
    Arr.push_back(to_string(A[i]));}
    
    if (check==false)   return to_string(0);
    sort(Arr.begin(), Arr.end(), compare);
    
    for(int i=0; i<Arr.size(); i++)
    ans = ans + Arr[i];
    
    return ans;
}
