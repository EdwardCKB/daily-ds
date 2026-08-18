import duckdb

result = duckdb.sql("""
    SELECT *, 
        RANK() OVER(PARTITION BY Type ORDER BY RI DESC) AS ri_rank
    FROM 'ml/data/glass.csv' 
    ORDER BY Type, ri_rank
    -- NOTE: RANK() skips numbers after ties (e.g. 70, 73, 74...) — two tied
    -- rows both get the same rank, and the next distinct row jumps ahead by
    -- however many rows tied. DENSE_RANK() does the same job without skipping,
    -- worth using instead if skipped numbers would be a problem for your use case
""").df()

print(result)



