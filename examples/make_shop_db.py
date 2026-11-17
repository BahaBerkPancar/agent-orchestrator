"""Builds examples/data/shop.db (deterministic demo data). Run: python examples/make_shop_db.py"""

import sqlite3
from pathlib import Path

path = Path(__file__).parent / "data" / "shop.db"
path.unlink(missing_ok=True)
con = sqlite3.connect(path)
con.executescript("""
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, city TEXT);
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, qty INTEGER, day TEXT);
INSERT INTO customers VALUES (1,'Ada Lovelace','ada@example.com','London'),
  (2,'Alan Turing','alan@example.com','Manchester'),(3,'Grace Hopper','grace@example.com','New York'),
  (4,'Linus Torvalds','linus@example.com','Portland');
INSERT INTO products VALUES (1,'Keyboard',49.5),(2,'Monitor',189.0),(3,'USB cable',4.25),(4,'Laptop stand',32.0);
INSERT INTO orders VALUES (1,1,2,1,'2026-10-01'),(2,1,1,2,'2026-10-03'),(3,2,3,10,'2026-10-04'),
  (4,3,2,2,'2026-10-07'),(5,3,4,1,'2026-10-09'),(6,4,1,1,'2026-10-11'),(7,3,1,3,'2026-10-15'),
  (8,2,4,2,'2026-10-20');
""")
con.commit()
con.close()
