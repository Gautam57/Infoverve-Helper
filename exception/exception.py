import sys
from exception.logger import logging

def error_message_details(error,error_detail:sys):
    #exc_tb gives us the info on which line the error occured and witch file the error occured
    _,_,exc_tb=error_detail.exc_info() 
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))

    return error_message
    

class CustomException(Exception):
    def __init__(self, error_message,error_details:sys):
        super().__init__(error_message) # Call the parent Exception class
        self.error_message=error_message_details(error_message,error_detail=error_details)

    def __str__(self):
        return self.error_message    