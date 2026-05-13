import pandas as pd

from impala.dbapi import connect

# Connect to Impala
conn = connect(
    host='10.168.49.12',
    port=21050,  # Default Impala port
    database='roam352_report_digi',
    user='unified',
    password='unified'
)

# Create cursor
cursor = conn.cursor()

# Execute query
cursor.execute(
    '''
    SELECT 
        *
    FROM 
        roam352_report_digi.data_em 
    WHERE
        par_date in (20240229)
        AND mcc_ref in (505, 502)  
    '''
)

# Get columns name
columns = [col[0] for col in cursor.description]

# Fetch results
results = pd.DataFrame(cursor.fetchall(), columns=columns)

results.to_csv('./data/20240229__transaction.csv', index=False)

# Close connection
cursor.close()
conn.close()

