from tika import parser

parsed_pdf = parser.from_file("../data/apolices/tokio_outubro_2024.pdf")

data = parsed_pdf['content']

print(data)