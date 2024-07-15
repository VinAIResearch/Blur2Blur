# from pdf2image import convert_from_path

# # Specify the PDF file path
# pdf_file = 'Blur2Blur_Paper.pdf'

# # Convert the first page of the PDF to an image
# images = convert_from_path(pdf_file, first_page=1, last_page=1)

# # Save the first page image to a file or display it
# if images:
#     first_page_image = images[0]
#     first_page_image.save('zip_blur2blur.png', 'PNG')  # Save the image as a PNG file
#     first_page_image.show()  # Display the image using the default image viewer

import fitz  # PyMuPDF

# Specify the PDF file path
pdf_file = 'Blur2Blur_Paper.pdf'

# Open the PDF file
pdf_document = fitz.open(pdf_file)

# Get the first page
first_page = pdf_document.load_page(0)

# Convert the first page to an image
image = first_page.get_pixmap()

# Save the first page image to a file or display it
image.save('zip_blur2blur.png')
# image.show()
