# importing module
import logging


# Create and configure logger
logging.basicConfig(filename="log.txt", format="%(asctime)s %(message)s",filemode='w')
 
# Creating an object
logger = logging.getLogger()

logger.debug("Harmless debug Message")
logger.info("Just an information")
logger.warning("Its a Warning")
logger.error("Did you try to divide by zero")
logger.critical("Internet is down")