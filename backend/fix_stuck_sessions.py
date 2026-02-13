"""Mark crashed batch search sessions as failed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_helper import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute(
    "SELECT Id, QueryImageName, CurrentProgress FROM ImageSearchHistory "
    "WHERE Status = 'in_progress' AND SearchType LIKE '%Batch%'"
)
rows = cursor.fetchall()
print(f"Found {len(rows)} stuck batch sessions:")
for r in rows:
    print(f"  #{r[0]}: {r[1]} - {r[2]}")

cursor.execute(
    "UPDATE ImageSearchHistory SET Status = 'failed' "
    "WHERE Status = 'in_progress' AND SearchType LIKE '%Batch%'"
)
conn.commit()
print(f"Updated {cursor.rowcount} sessions to 'failed'")
cursor.close()
conn.close()
