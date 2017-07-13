import cherrypy

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

    

    
if __name__ == '__main__':
    cherrypy.quickstart(API(), '/', config)
