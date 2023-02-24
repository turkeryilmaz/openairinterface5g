import xml.etree.ElementTree as ET

def merge_xml_tree(xml_file, tree_file):
    # Parse the XML and tree files
    xml_tree = ET.parse(xml_file)
    tree_tree = ET.parse(tree_file)

    # Get the root element of the XML tree
    root = xml_tree.getroot()

    # Iterate over the elements in the tree tree
    for tree_element in tree_tree.iter():
        # Find the corresponding element in the XML tree
        xml_element = root.find(tree_element.tag)

        # If the element exists in the XML tree, add the 'rw' or 'ro' attribute
        if xml_element is not None:
            xml_element.set('access', tree_element.get('rw', 'ro'))

    # Return the merged XML tree as a string
    return ET.tostring(root)

# Example usage
xml_file = 'xml/gnodeb.xml'
tree_file = 'xml/gnodeb.tree'
merged_xml = merge_xml_tree(xml_file, tree_file)
print(merged_xml)



