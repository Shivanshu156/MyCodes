create materialized view home as SELECT b.icon,b.title,b.category,p.price, b.rating,b.url FROM
  (select a.icon,a.title,c.category,a.rating,a.url from app a join categories c on a.url = c.app_url) b join
  pricing_plans p on p.app_url=b.url;

alter view home
add column duration text;



alter table pricing_plans alter column price type float using(price::double precision);

update pricing_plans set price=replace(price,'Free','0');

  CREATE INDEX idx_1
  ON  home(rating);

  CREATE INDEX idx_2
  ON  home(category);

  CREATE INDEX idx_3
  ON  home(price);

  CREATE INDEX idx_4
  ON  home(title);

alter table app add constraint c1
unique(url,title);

alter table app alter column
url set not null;

alter table categories alter column
app_url set not null;

alter table pricing_plans add constraint c2
unique(id,app_url);


CREATE OR REPLACE FUNCTION rec_insert()
  RETURNS trigger AS
$$
BEGIN
         INSERT INTO home(icon,title,price,rating,url)
         VALUES(NEW.icon,NEW.title,NEW.price,NEW.rating,NEW.url);

    RETURN NEW;
END;
$$
LANGUAGE 'plpgsql';

CREATE TRIGGER ins_same_rec
  AFTER INSERT
  ON app
  FOR EACH ROW
  EXECUTE PROCEDURE rec_insert();


CREATE OR REPLACE FUNCTION rec_insert_ctg()
    RETURNS trigger AS
  $$
  BEGIN
          update  home set category = NEW.category
          where NEW.app_url = home.app_url and home.category = NULL;

      RETURN NEW;
  END;
  $$
  LANGUAGE 'plpgsql';

  CREATE TRIGGER ins_same_rec_ctg
    AFTER INSERT
    ON categories
    FOR EACH ROW
    EXECUTE PROCEDURE rec_insert_ctg();


SELECT * FROM home where category = 'Productivity' ORDER by rating desc, price asc;

SELECT * FROM home
    where lower(title) Like lower('%site%') ORDER by rating desc ;

SELECT * FROM home 
   where lower(title) Like lower('%hu%') and category = 'Marketing' order by rating desc;
