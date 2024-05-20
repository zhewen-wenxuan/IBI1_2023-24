# DOM
import xml.dom.minidom
import time
from collections import defaultdict
import matplotlib.pyplot as plt

start_time = time.time()
# Load and parse the XML file
dom = xml.dom.minidom.parse('/Users/xuanzhewen/code/IBI1_2023-24/Practical14/go_obo.xml')
# Extract the terms and their namespaces
terms = dom.getElementsByTagName('term')
namespace_count = defaultdict(int)

for term in terms:
    namespace = term.getElementsByTagName('namespace')[0].childNodes[0].data
    namespace_count[namespace] += 1

end_time = time.time()
dom_time = end_time - start_time

print(f"DOM Parsing Time: {dom_time:.2f} seconds")

# Plotting the results
namespaces = list(namespace_count.keys())
counts = list(namespace_count.values())

plt.bar(namespaces, counts, color=['blue', 'green', 'red'])
plt.xlabel('Ontology')
plt.ylabel('Number of Terms')
plt.title('Number of Terms per Ontology (DOM)')
plt.show()

# Output the counts for each ontology
for namespace, count in namespace_count.items():
    print(f"{namespace}: {count}")

# SAX
import xml.sax
import time
from collections import defaultdict
import matplotlib.pyplot as plt
class GOSaxHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.current_tag = ""
        self.namespace_count = defaultdict(int)

    def startElement(self, tag, attributes):
        self.current_tag = tag

    def endElement(self, tag):
        self.current_tag = ""

    def characters(self, content):
        if self.current_tag == "namespace":
            self.namespace_count[content] += 1

start_time = time.time()

# Create an XMLReader
parser = xml.sax.make_parser()
# Turn off namespace processing
parser.setFeature(xml.sax.handler.feature_namespaces, 0)

# Override the default ContextHandler
Handler = GOSaxHandler()
parser.setContentHandler(Handler)

parser.parse("/Users/xuanzhewen/code/IBI1_2023-24/Practical14/go_obo.xml")

end_time = time.time()
sax_time = end_time - start_time

print(f"SAX Parsing Time: {sax_time:.2f} seconds")

# Plotting the results
namespaces = list(Handler.namespace_count.keys())
counts = list(Handler.namespace_count.values())

plt.bar(namespaces, counts, color=['blue', 'green', 'red'])
plt.xlabel('Ontology')
plt.xticks(rotation=90)
plt.ylabel('Number of Terms')
plt.title('Number of Terms per Ontology (SAX)')
plt.show()

# Output the counts for each ontology
for namespace, count in Handler.namespace_count.items():
    print(f"{namespace}: {count}")

# In my laptop, SAX is faster than DOX
