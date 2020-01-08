#TODO test large bulk 100 300 1000 3000
#TODO add image dedup perceptual hash or cnn embedding
#TODO add cross val

from datetime import datetime, timedelta
from google_images_download import google_images_download

bulks = [100, 300]
for b in bulks:
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": "chien",
                "limit": b,
                "chromedriver": "/opt/chromedriver-79.0.3945.36/chromedriver",
    #             "time_range": '{"time_min":"02/01/2018", "time_max":"03/01/2018"}', # MM/DD/YYYY
                "print_urls": True,
                "output_directory": "./downloads/buld"+str(b),
                "no_directory": True}
    paths = response.download(arguments)


# start = "01/01/2018"
# for i in range(11):
#     end = str(int(start[:2])+1).zfill(2)+start[2:]
#     print(f"{i}th sample has range {start=} - {end=}")
#     start = end
