""" Demo Python Flask Application """

import os
import sys

import psycopg2

from flask import Flask, render_template, redirect, request

# Connect to the database
db = "host=localhost dbname=dbms_project1 user=postgres password=postgres"
conn = psycopg2.connect(db)
cur = conn.cursor()

app = Flask(__name__)

@app.route("/admin/access")
def admin():
    return render_template("admin.html")

@app.route("/add/app",methods=['GET'])
def add1():

    return render_template("add.html")

@app.route("/delete/app",methods=['POST'])
def delete1():

    return render_template("delete.html")

@app.route("/update/app",methods=['POST'])
def update1():

    return render_template("add.html")



@app.route("/add/app/result", methods=['POST'])
def add():
    if(request.method=='POST'):

        title = request.form.get["addtitle"]
        developer = request.form.get["adddeveloper"]

        dev_link = request.form.get["adddeveloperlink"]
        url = request.form.get['addurl']
        icon = request.form.get["addicon"]
        kb= request.form.get["addkb"]
        rating = request.form.get["addrating"]
        review = request.form.get["addreview"]
        descr = request.form.get["adddescription"]
        sdescr = request.form.get["addshortdescription"]
        categ = request.form.get["addcategory"]
        id = request.form.get["addid"]
        price = request.form.get["addprice"]
        hint = request.form.get["addhint"]
        fea = request.form.get["addfeature"]
        cur.execute(
        """ Insert into app(url,title,developer,developer_link,icon,rating,reviews_count,Description
        short_desscription,key_benefits,pricing,pricing_hints)
        Values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ;
        """,(url,title,developer,dev_link,icon,rating,review,descr,sdescr,kb,price,hint))
        result1 = cur.statusmessage
        cur.execute(
        """ Insert into categories(app_url,category)
        Values(%s,%s)
        ;
        """,(url,categ))
        result2 = cur.statusmessage
        cur.execute(
        """ Insert into plan_features(plan_id,app_url,feature)
        Values(%s,%s,%s)
        ;
        """,(id,url,fea))
        result3=cur.statusmessage
        cur.execute(
        """ Insert into pricing_plans(id,app_url,title,price,hint)
        Values(%s,%s,%s,%s,%s)
        ;
        """,(id,url,title,price,hint))
        result4=cur.statusmessage
        print(result1,'result1',result2,'result2',result3,'result3',result4,'result4')
        return render_template("result.html")

    else:
        return render_template("result.html")

@app.route("/delete/app/result", methods=['POST'])
def delete():
    if(request.method=='POST'):

        url = request.form.get['deleteurl']
        cur.execute(
        """ delete from app
         where url=%s
        ;
        """,(url,))
        result1 = cur.statusmessage
        cur.execute(
        """ delete from categories
        where url=%s
        ;
        """,(url,))
        result2 = cur.statusmessage
        cur.execute(
        """ delete from plan_features
        where url=%s
        ;
        """,(url,))
        result3=cur.statusmessage
        cur.execute(
        """ delete from pricing_plans
        where url=%s
        ;
        """,(url,))
        result4=cur.statusmessage
        print(result1,'result1',result2,'result2',result3,'result3',result4,'result4')
        return render_template("deleteresult.html")

    else:
        return render_template("deleteresult.html")



@app.route("/")
def root():
    cur.execute(
        """SELECT * FROM home ORDER by rating desc, price asc LIMIT 25;"""
        )
    rows = cur.fetchall()
    return render_template("home.html", rows=rows)

@app.route("/<ctg>")
def Category(ctg):
    # print (ctg,'is Category .................')
    cur.execute(
        " SELECT * FROM home where category = %s ORDER by rating desc, price asc;",(ctg,))
    rows = cur.fetchall()
    # print ('total rows fetched =', len(rows))

    return render_template("categories.html", rows=rows,ctg = ctg)


@app.route("/search", methods=['POST'])
def search():
    name = request.form["sinput"]
    name='%'+name+'%'
    sort='1'
    sort = request.form['sort']


    if (sort=='1'):
        cur.execute("""SELECT * FROM home
        where lower(title) Like lower(%s) ORDER by rating desc ;""",(name,))
        rows = cur.fetchall()
        print("order by rating desc")
        return render_template("search.html", rows=rows)

    elif sort =='2':
        cur.execute("""SELECT * FROM home
        where lower(title) Like lower(%s) ORDER by price asc ;""",(name,))
        rows = cur.fetchall()
        return render_template("search.html", rows=rows)

    elif sort =='3':
        cur.execute("""SELECT * FROM home
        where lower(title) Like lower(%s) ORDER by price desc ;""",(name,))
        rows = cur.fetchall()
        return render_template("search.html", rows=rows)

    else:
        cur.execute("""SELECT * FROM home
        where lower(title) Like lower(%s) ;""",(name,))
        rows = cur.fetchall()
        return render_template("search.html", rows=rows)

@app.route("/<ctg>/searching", methods=['POST'])
def searching(ctg):
    name = request.form["sinput1"]
    name='%'+name+'%'
    print(name," is name in category ",ctg)
    sort1='1'
    sort1 = request.form['sort1']
    print(name, "in category",ctg, "is of value", sort1 )
    if sort1 == '1':
        cur.execute("""SELECT * FROM home
        where lower(title) Like lower(%s) and category = %s order by rating desc;""",(name,ctg))
        rows = cur.fetchall()
        return render_template("categories.html", rows=rows,ctg=ctg)

    elif sort1 == '2':
        cur.execute("""SELECT * FROM home
        where lower(title) Like lower(%s) and category = %s order by price asc;""",(name,ctg))
        rows = cur.fetchall()
        return render_template("categories.html", rows=rows,ctg=ctg)

    elif sort1 == '3':
        cur.execute("""SELECT * FROM home
        where lower(title) Like lower(%s) and category = %s order by price desc;""",(name,ctg))
        rows = cur.fetchall()
        return render_template("categories.html", rows=rows,ctg=ctg)

    else:
        cur.execute("""SELECT * FROM home
        where lower(title) Like lower(%s) and category = %s;""",(name,ctg))
        rows = cur.fetchall()
        return render_template("categories.html", rows=rows,ctg=ctg)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
