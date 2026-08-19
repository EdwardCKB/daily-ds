import duckdb

# PARTITION BY and Dense() vs Dense_rank()
ses_1 = duckdb.sql("""
    SELECT *, 
        RANK() OVER(PARTITION BY Type ORDER BY RI DESC) AS ri_rank
    FROM 'ml/data/glass.csv' 
    ORDER BY Type, ri_rank
    -- NOTE: RANK() skips numbers after ties (e.g. 70, 73, 74...) — two tied
    -- rows both get the same rank, and the next distinct row jumps ahead by
    -- however many rows tied. DENSE_RANK() does the same job without skipping,
    -- worth using instead if skipped numbers would be a problem for your use case
""").df()

ses_2 = duckdb.sql("""
    WITH ranked AS (
        SELECT *,
            RANK() OVER (PARTITION BY Type ORDER BY RI DESC) as rn
        FROM 'ml/data/glass.csv')
    SELECT *
    FROM ranked
    WHERE rn = 1
    ORDER BY Type
""").df()

# Running sum
ses_3 = duckdb.sql("""
    SELECT *,
        SUM(Fe) OVER (ORDER BY RI ASC) AS running_sum
    FROM 'ml/data/glass.csv'
    ORDER BY RI ASC
""").df()

# Moving Average 
ses_4 =duckdb.sql("""
    SELECT *,
    AVG(Fe) OVER (ORDER BY RI ASC ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3
    -- explicit 3-row frame: 2 rows before + current row.
    -- Handles the start-of-data edge case gracefully — row 0 has
    -- only itself (avg of 1 value), row 1 has 2 rows available,
    -- only row 2 onward gets a genuine full 3-row window
    FROM 'ml/data/glass.csv'
    ORDER BY RI ASC 
    -- this outer ORDER BY is what actually guarantees rows print in RI
    -- order — without it, output row order isn't guaranteed even though
    -- the calculation itself was still done correctly in RI sequence
""").df()
print(ses_4)

