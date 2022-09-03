#include <iostream>
using namespace std;

struct node
{
	bool flag;
	long int value;
	node *rgt, *down;

	int r;
	int c;
};

node *l = NULL;

node *AddLink(node *l, int i, int j)
{
	node *p, *q, *start, *pc, *qc;

	if (l == NULL)
	{
		node *n, *s, *t;
		n = new node;
		n->flag = true;
		n->value = 0;
		n->down = n;
		n->rgt = n;
		n->c = -1;
		n->r = -1; // cout<<"n->u.c = "<<n->c;
		s = new node;
		s->flag = true;
		s->value = 0;
		s->c = -1;
		s->r = i;
		s->down = n;
		s->rgt = s; //cout<<"s->u.c = "<<s->c<<endl;
		n->down = s;
		t = new node;
		t->flag = true;
		t->value = 0;
		t->down = t;
		t->rgt = n;
		t->r = -1;
		t->c = j; //cout<<"t->u.r = "<<t->r<<endl;
		n->rgt = t;
		l = n;
	}
	//for row wise start
	start = l;
	q = l;
	p = q->down;
	while (q->r < i && p != start)
	{
		bool cond = true;
		if (p->r == i)
		{
			if (p->rgt == p)
			{
				node *g = new node;
				g->flag = false;
				g->r = i;
				g->c = j;
				g->value = 1;
				p->rgt = g;
				g->rgt = p;
				qc = l;
				pc = qc->rgt;
				while (qc->c < j && pc != start)
				{
					bool condc = true;
					if (pc->c == j)
					{
						if (pc->down == pc)
						{
							pc->down = g;
							g->down = pc; //cout<<"added easily without traversing";
							break;
						}

						node *ac, *bc;
						ac = pc;
						bc = pc->down;
						while (ac->r < i && bc != pc)
						{
							if (bc->r == i)
							{
								break;
							}
							if (bc->r > i)
							{
								g->down = bc;
								ac->down = g;
							}
							if (bc->r < i && bc->down == pc)
							{
								bc->down = g;
								g->down = pc;
							}

							ac = bc;
							bc = ac->down;
						}
					}
					if (pc->c > j)
					{
						node *s1 = new node;
						s1->flag = true;
						s1->value = 0;
						s1->rgt = pc;
						qc->rgt = s1;
						s1->down = s1;
						s1->r = -1;
						s1->c = j;
						condc = false;
						pc = s1;
					}
					if (pc->c < j && pc->rgt == start)
					{
						node *s3 = new node;
						s3->flag = true;
						s3->value = 0;
						s3->rgt = start;
						s3->down = s3;
						s3->r = -1;
						s3->c = j;
						pc->rgt = s3;
					}

					if (condc == true)
					{
						qc = pc;
						pc = qc->rgt;
					}
				}
				////////////////linking with column head node end
				break;
			}

			node *a, *b;
			a = p;
			b = a->rgt;

			while (a->c < j && b != p)
			{
				if (b->c == j)
				{ //cout<<"b++";
					b->value = b->value + 1;
					break;
				}
				if (b->c > j)
				{
					node *h = new node;
					h->flag = false;
					h->c = j;
					h->r = i;
					h->value = 1;
					h->rgt = b;
					a->rgt = h;

					qc = l;
					pc = qc->rgt;
					while (qc->c < j && pc != start)
					{
						bool condc = true;
						if (pc->c == j)
						{
							if (pc->down == pc)
							{
								pc->down = h;
								h->down = pc;
								break;
							}

							node *ac, *bc;
							ac = pc;
							bc = ac->down;
							while (ac->r < i && bc != pc)
							{
								if (bc->r == i)
								{
									break;
								}
								if (bc->r > i)
								{
									h->down = bc;
									ac->down = h;
								}
								if (bc->r < i && bc->down == pc)
								{
									bc->down = h;
									h->down = pc;
								}

								ac = bc;
								bc = ac->down;
							}
						}
						if (pc->c > j)
						{
							node *s = new node;
							s->flag = true;
							s->value = 0;
							s->rgt = pc;
							qc->rgt = s;
							s->down = s;
							s->r = -1;
							s->c = j;
							condc = false;
							pc = s;
						}
						if (pc->c < j && pc->rgt == start)
						{
							node *s = new node;
							s->flag = true;
							s->value = 0;
							s->rgt = start;
							s->down = s;
							s->r = -1;
							s->c = j;
							pc->rgt = s;
						}

						if (condc == true)
						{
							qc = pc;
							pc = qc->rgt;
						}
					}
				}
				if (b->c < j && b->rgt == p)
				{
					node *h = new node;
					h->flag = false;
					h->c = j;
					h->r = i;
					h->value = 1;
					h->rgt = p;
					b->rgt = h;
					//							 b=h;

					qc = l;
					pc = qc->rgt;
					while (qc->c < j && pc != start)
					{
						bool condc = true;
						if (pc->c == j)
						{
							if (pc->down == pc)
							{
								pc->down = h;
								h->down = pc;
								break;
							}

							node *ac, *bc;
							ac = pc;
							bc = ac->down;
							while (ac->r < i && bc != pc)
							{
								if (bc->r == i)
								{
									break;
								}
								if (bc->r > i)
								{
									h->down = bc;
									ac->down = h;
								}
								if (bc->r < i && bc->down == pc)
								{
									bc->down = h;
									h->down = pc;
								}

								ac = bc;
								bc = ac->down;
							}
						}
						if (pc->c > j)
						{
							node *s = new node;
							s->flag = true;
							s->value = 0;
							s->rgt = pc;
							qc->rgt = s;
							s->down = s;
							s->r = -1;
							s->c = j;
							condc = false;
							pc = s;
						}
						if (pc->c < j && pc->rgt == start)
						{
							node *s = new node;
							s->flag = true;
							s->value = 0;
							s->rgt = start;
							s->down = s;
							s->r = -1;
							s->c = j;
							pc->rgt = s;
						}

						if (condc == true)
						{
							qc = pc;
							pc = qc->rgt;
						}
					}
					/////////////////////////////////////////////////////////////////////////////////								 ////////////////////

					b = h;
				}

				a = b;
				b = a->rgt;
			}
		}
		if (p->r > i)
		{
			node *s2 = new node;
			s2->flag = true;
			s2->value = 0;
			s2->down = p;
			q->down = s2;
			s2->rgt = s2;
			s2->c = -1;
			s2->r = i;
			cond = false;
			p = s2;
		}
		if (p->r < i && p->down == start)
		{
			node *s3 = new node;
			s3->flag = true;
			s3->value = 0;
			s3->down = start;
			s3->rgt = s3;
			s3->c = -1;
			s3->r = i;
			p->down = s3;
		}

		if (cond == true)
		{
			q = p;
			p = q->down;
		}
	}
	return (l);
}

node *DeleteLink(node *l, int i, int j)
{
	if (l == NULL)
		return (l);
	node *pd, *qd, *start, *pcd, *qcd;
	start = l;
	qd = l;
	pd = qd->down;

	while (qd->r < i && pd != start)
	{
		if (pd->r == i)
		{
			node *ad, *bd;
			ad = pd;
			bd = ad->rgt;
			while (ad->c < j && bd != pd)
			{
				if (bd->c == j) //Node found
				{
					if (bd->value == 1)
					{
						ad->rgt = bd->rgt;
						bd = ad->rgt;
						if (pd->rgt == pd)
						{
							qd->down = pd->down;
							pd = qd->down;
						}

						qcd = start;
						pcd = qcd->rgt; //coming on deletion from column node
						while (pcd->c < j)
						{
							qcd = qcd->rgt;
							pcd = qcd->rgt;
						}

						node *acd, *bcd;
						if (pcd->c == j)
						{
							acd = pcd;
							bcd = acd->down;
						}
						while (acd->r < i && bcd != pcd)
						{
							if (bcd->r == i)
							{
								bcd->value = 0;
								acd->down = bcd->down;
								bcd = acd->down;
								break;
							}
							acd = acd->down;
							bcd = acd->down;
						}
						if (pcd->down == pcd)
						{
							qcd->rgt = pcd->rgt;
							pcd = qcd->rgt;
						}
						break;
					}
					if (bd->value > 1)
					{
						bd->value = bd->value - 1;
					}
				}
				if (bd->c > j)
					break; // Node does not exist
				if (bd->c < j && bd->rgt == start)
					break; // Node does not exist
				ad = bd;
				bd = ad->rgt;
			}
		}
		if (pd->r > i)
			break;
		if (pd->r < i && pd->down == start)
			break;
		qd = qd->down;
		pd = qd->down;
	}

	return (l);
}

int RetrieveValue(node *l, int i, int j)
{
	cout << "aa gye loop me";
	if (l == NULL)
		return 0;
	node *p, *q, *start;
	int value1;
	start = l;
	q = l;
	p = q->down;
	cout << "before while";
	while (q->r < i && p != start)
	{
		cout << "p->u.r = " << p->r << endl;

		if (p->r == i)
		{
			node *a3, *b3;
			a3 = p;
			b3 = a3->rgt;
			cout << "b3->u.c = " << b3->c << endl;
			cout << "a3->u.c = " << a3->c << endl;
			while (a3->c < j && b3 != p)
			{
				cout << "while me aa gya";
				if (b3->c == j)
				{
					cout << "node found";
					value1 = b3->value;
				}
				if (b3->c > j)
				{
					value1 = 0;
					break;
				}
				if (b3->c < j && b3->rgt == p)
				{
					value1 = 0;
					break;
				}
				a3 = b3;
				b3 = a3->rgt;
			}
		}
		if (p->r > i)
		{
			value1 = 0;
			break;
		}
		if (p->r < i && p->down == start)
		{
			value1 = 0;
			break;
		}
		q = p;
		p = p->down;
	}
	return value1;
}

int RetrieveRowSum(node *l, int i, int k)
{
	int sum4 = 0, count4 = 0;
	node *p4, *q4, *start;
	start = l;
	q4 = l;
	p4 = q4->down;
	while (q4->r < i && p4 != start)
	{
		if (p4->r == i)
		{
			if (p4->rgt == p4)
				sum4 = 0;
			node *a4, *b4;
			a4 = p4;
			b4 = a4->rgt;
			while (b4 != p4 && count4 < k)
			{
				sum4 = sum4 + b4->value;
				count4++;
				a4 = b4;
				b4 = a4->rgt;
			}
		}
		if (p4->r > i)
			sum4 = 0;
		if (p4->r < i && p4->down == start)
			sum4 = 0;

		q4 = p4;
		p4 = q4->down;
	}
	return sum4;
}

int RetrieveColSum(node *l, int j, int k)
{
	int sum5 = 0, count5 = 0;
	node *p5, *q5, *start;
	start = l;
	q5 = l;
	p5 = q5->rgt;
	while (q5->c < j && p5 != start)
	{
		if (p5->c == j)
		{
			if (p5->down == p5)
			{
				sum5 = 0;
				break;
			}
			node *a5, *b5;
			a5 = p5;
			b5 = p5->down;
			while (b5 != p5 && count5 < k)
			{
				sum5 = sum5 + b5->value;
				count5 = count5 + 1;
				a5 = b5;
				b5 = a5->down;
			}
		}
		if (p5->c > j)
			sum5 = 0;
		if (p5->c < j && p5->rgt == start)
			sum5 = 0;

		q5 = p5;
		p5 = q5->rgt;
	}
	return sum5;
}

int MultiplyVector(node *l, int n, int a[])
{
	node *p, *q;
	p = l->down;
	int b[n];
	for (int k = 0; k < n; k++)
	{
		int val = 0;
		if (p->r == k)
		{
			q = p->rgt;
			for (int j = 0; j < n && q != p; j++)
			{
				if (q->c == j)
				{
					val = val + q->value * a[j];

					q = q->rgt;
				}
			}
			b[k] = val;
			p = p->down;
		}
		else
			b[k] = 0;
		cout << b[k] << " ";
	}
}

int main()
{
	int num, fun, a, b, n;
	//cout<<"Enter the number of operations";
	cin >> num;
	for (int i = 0; i < num; i++)
	{ //cout<<i+1<<". ";
		cin >> fun;

		switch (fun)
		{

		case 1: /*cout<<"Entre i and j for addlink";*/
			cin >> a >> b;
			l = AddLink(l, b, a);
			break;
		case 2: /*cout<<"Entre i and j for delete link";*/
			cin >> a >> b;
			l = DeleteLink(l, b, a);
			break;
		case 3: /*cout<<"Entre i and j for retrievevalue";*/
			cin >> a >> b;
			cout << RetrieveValue(l, b, a) << endl;
			break;
		case 4: /*cout<<"Entre i and k for retrieverowsumuptokthcolumn";*/
			cin >> a >> b;
			cout << RetrieveRowSum(l, a, b) << endl;
			break;
		case 5: /*cout<<"Entre j and k for retrievecolumnsumuptokthrow"; */
			cin >> a >> b;
			cout << RetrieveColSum(l, a, b) << endl;
			break;
		case 6:
			cin >> n;
			int a[n];
			for (int k = 0; k < n; k++)
				cin >> a[k];
			MultiplyVector(l, n, a);
			cout << endl;
			break;
		}
	}
}
