# USAGE
# python search.py --dataset images --shelve db.shelve --query images/84eba74d-38ae-4bf6-b8bd-79ffa1dad23a.jpg

# import the necessary packages
from PIL import Image
import imagehash
import argparse
import shelve
import  os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to dataset of images")
ap.add_argument("-s", "--shelve", required = True,
	help = "output shelve database")
args = vars(ap.parse_args())

# open the shelve database
db = shelve.open(args["shelve"])

# load the query image, compute the difference image hash, and
# and grab the images from the database that have the same hash
# value

for file in os.listdir(args["dataset"]):
	print "now check", file, "..."
	if(file not in os.listdir(args["dataset"])):
		print file, "has already been deleted"
		pass
	else:
		query = Image.open(args["dataset"] + "/" + file)
		h = str(imagehash.dhash(query))
		filenames = db[h]
		print "Found %d images" % (len(filenames))

		# loop over the images
		reserve = filenames[0]
		for filename in filenames:
			if(filename != reserve):
				print filename, "deleted"
				if (filename in os.listdir(args["dataset"])):
					os.remove(args["dataset"] + "/" + filename)
				else:
					print filename, "has already been deleted"


			# image = Image.open(args["dataset"] + "/" + filename)
			# image.show()

		# close the shelve database
db.close()