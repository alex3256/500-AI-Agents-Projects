import argparse
import os
import re
import sqlite3
from urllib.parse import quote

from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI


def clean_output(text: str) -> str:
    """Remove common Markdown formatting for terminal display."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'__(.+?)__', r'\1', text)       # __bold__
    text = re.sub(r'`(.+?)`', r'\1', text)          # `code`
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # ## headings
    return text


load_dotenv()


def create_demo_database(db_path: str):
    """Creates a demo e-commerce SQLite database for testing."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            country TEXT,
            created_at DATE DEFAULT CURRENT_DATE
        );
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER NOT NULL,
            total REAL NOT NULL,
            order_date DATE DEFAULT CURRENT_DATE
        );
        INSERT OR IGNORE INTO customers VALUES
            (1,'Alice Johnson','alice@example.com','USA','2024-01-15'),
            (2,'Bob Smith','bob@example.com','UK','2024-02-20'),
            (3,'Carlos Lima','carlos@example.com','Brazil','2024-03-10'),
            (4,'Diana Prince','diana@example.com','USA','2024-01-05');
        INSERT OR IGNORE INTO products VALUES
            (1,'Laptop Pro','Electronics',1299.99,45),
            (2,'Wireless Mouse','Electronics',29.99,200),
            (3,'Python Book','Books',49.99,120),
            (4,'Standing Desk','Furniture',599.99,15);
        INSERT OR IGNORE INTO orders VALUES
            (1,1,1,1,1299.99,'2024-04-01'),
            (2,1,2,2,59.98,'2024-04-01'),
            (3,2,3,1,49.99,'2024-04-05'),
            (4,3,4,1,599.99,'2024-04-10'),
            (5,4,1,1,1299.99,'2024-04-12'),
            (6,2,2,3,89.97,'2024-04-15');
    """)
    conn.commit()
    conn.close()


def mysql_uri() -> str:
    """Build a MySQL URI from environment variables: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME.
    Raises ValueError if DB_HOST is not set.
    """
    host = os.environ.get("DB_HOST")
    if not host:
        raise ValueError("DB_HOST not set in .env — cannot connect to MySQL")
    port = os.environ.get("DB_PORT", "3306")
    user = os.environ.get("DB_USER", "root")
    password = quote(os.environ.get("DB_PASSWORD", ""))
    db_name = os.environ.get("DB_NAME", "")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"


def sqlite_uri(db_path: str, read_only: bool = True) -> str:
    abs_path = os.path.abspath(db_path)
    if read_only:
        return f"sqlite:///file:{quote(abs_path)}?mode=ro&uri=true"
    return f"sqlite:///{abs_path}"


def build_agent(db_path: str, read_only: bool = True, use_mysql: bool = False):
    if use_mysql:
        db = SQLDatabase.from_uri(mysql_uri())
    else:
        db = SQLDatabase.from_uri(sqlite_uri(db_path, read_only=read_only))
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        base_url="http://localhost:3000/v1",
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        verbose=False,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )
    return agent, db


def main():
    parser = argparse.ArgumentParser(description="SQL Query Agent")
    parser.add_argument("--db", default="demo.sqlite", help="SQLite database path (default: demo.sqlite)")
    parser.add_argument("--mysql", action="store_true", help="Connect to MySQL instead of SQLite (reads DB_HOST/DB_USER/DB_PASSWORD/DB_NAME from .env)")
    parser.add_argument("--question", help="Natural language question (omit for interactive)")
    parser.add_argument("--allow-write", action="store_true", help="Open the SQLite database read-write instead of read-only")
    args = parser.parse_args()

    if not args.mysql and args.db == "demo.sqlite" and not os.path.exists("demo.sqlite"):
        print("🏗️  Creating demo e-commerce database...")
        create_demo_database("demo.sqlite")

    agent, db = build_agent(args.db, read_only=not args.allow_write, use_mysql=args.mysql)

    if args.mysql:
        host = os.environ.get("DB_HOST", "?")
        db_name = os.environ.get("DB_NAME", "?")
        print(f"\n📊 Connected to MySQL: {host}/{db_name}")
    else:
        print(f"\n📊 Connected to: {args.db}")
        print(f"🔒 Mode: {'read-write' if args.allow_write else 'read-only'}")
    print(f"📋 Tables: {', '.join(db.get_usable_table_names())}\n")

    if args.question:
        print(f"❓ Question: {args.question}")
        result = agent.invoke({"input": args.question})
        print(f"\n✅ Answer: {clean_output(result['output'])}")
    else:
        print("💬 SQL Agent ready. Ask questions in natural language. Type 'quit' to exit.\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            result = agent.invoke({"input": question})
            print(f"\nAgent: {clean_output(result['output'])}\n")


if __name__ == "__main__":
    main()

