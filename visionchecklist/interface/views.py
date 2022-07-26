from django.shortcuts import render
from django.http import HttpResponseRedirect
from .models import Image
from django.shortcuts import get_object_or_404, render, get_list_or_404
from django.urls import reverse
from django.views import generic


class HomeView(generic.DetailView):
    model = Image
    template_name = 'interface/Home.html'

def ResultsView(request, mode, prop, angle):
    input_image = get_object_or_404(Image, image__contains='input', rotation_distribution=mode, rot_proportion=prop/100, angle=angle)
    output_image = input_image.output_set.all()[0]
    template_name = 'interface/Results.html'
    context = {
        'input_image': input_image,
        'output_image': output_image 
    }

    return render(request, template_name, context)


def select(request):
    if len(request.POST) < 4:
        return render(request, 'interface/Home.html', {'image':Image.objects.get(pk=1), 'error_message': "You need to select a configuration"})
    elif request.POST['angle'] == '0':
        return HttpResponseRedirect(reverse('interface:results', args=('R', 0, 0)))
    else:
        return HttpResponseRedirect(reverse('interface:results', args=(request.POST['mode'], request.POST['prop'], request.POST['angle'])))
    image = get_object_or_404(Image, pk=request.POST[''])
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'survey/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('survey:results', args=(question.id,)))