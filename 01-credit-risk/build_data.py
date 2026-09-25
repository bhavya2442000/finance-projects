"""Step 1: build the modeling table and vintage curves with DuckDB SQL."""
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent
DB = ROOT / "data" / "credit.duckdb"

con = duckdb.connect(str(DB))
for sql_file in sorted((ROOT / "sql").glob("*.sql")):
    print(f"running {sql_file.name}")
    con.execute(sql_file.read_text())

print(con.execute("""
    SELECT vintage_year, COUNT(*) AS loans, ROUND(AVG(is_bad), 4) AS bad_rate,
           ROUND(SUM(funded_amnt) / 1e9, 2) AS funded_bn
    FROM loans GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
print(con.execute("SELECT * FROM lgd_by_grade").df().round(3).to_string(index=False))
con.close()
