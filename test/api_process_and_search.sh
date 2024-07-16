curl \
    -X POST "http://127.0.0.1:6006/v1/process_and_search/" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "query_file=@../data/result_d20240509.xlsx"