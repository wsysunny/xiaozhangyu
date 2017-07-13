import cherrypy
import API.detect
import API.search

@cherrypy.expose
class API(object):
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        return {"key": "value"}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def search(self, **kwargs):
        result = search(**kwargs)
        return result
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def detect(self, img_data, img_type=1):
        img = Image()
        if img_type == '1':
            img.image_url = img_data
        if img_type == '2':
            img.image_file = img_data
        if img_type == '3':
            img.image_base64 = img_data
        else:
            pass  # TODO
        result = detect(img)
        return result

    

    
if __name__ == '__main__':
    cherrypy.quickstart(API(), '/', config)
