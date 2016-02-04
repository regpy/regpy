{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   Functions
   ---------
   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}

   {% for item in functions %}
   .. autofunction:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   Classes
   -------
   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}

   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   Exceptions
   ----------
   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}

   {% for item in exceptions %}
   .. autoexception:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
