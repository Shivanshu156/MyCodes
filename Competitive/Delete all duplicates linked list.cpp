ListNode* Solution::deleteDuplicates(ListNode* A) {
    
    ListNode *current, *prev, *head; 
    head = new ListNode(-1);
    prev = head;    current = A;    head->next = current;

    while(current!=NULL)    
    {   if(current->next!=NULL)
        {
            if(current->val!=current->next->val)
            {   prev = current;
                current = current->next;
            }
            else
            {
                int value = current->val;
                while(current->val == value)
                {prev->next = current->next;
                free(current);
                current = prev->next;
                if(current == NULL) break;
                }
            
            }
            
        }
        else current = current->next;
        
    }

    if(head->next !=NULL)
    return head->next;
    else return NULL;
    
}