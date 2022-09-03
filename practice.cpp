#include<iostream>
#include<string>
using namespace std;

struct ListNode {
      int val;
      ListNode *next;
      ListNode(int x) : val(x), next(NULL) {}
  };

int main()
{   
    string str;
    cin>>str;
    cout<<str;
}